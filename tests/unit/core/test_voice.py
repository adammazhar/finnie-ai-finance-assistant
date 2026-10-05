"""Voice: speech to text for questions, and answers prepared for reading aloud."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from src.core.guardrails import SHORT_DISCLAIMER
from src.core.voice import VOCABULARY, VoiceError, speakable_text, transcribe, voice_available


class FakeTranscriptions:
    def __init__(self, text="What is an ETF?", error=None):
        self.text, self.error, self.calls = text, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


def client(**kwargs):
    transcriptions = FakeTranscriptions(**kwargs)
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions)), transcriptions


@pytest.fixture
def settings(make_settings):
    return make_settings(openai_api_key="sk-test-voice-key")


def without_key(settings, key=None):
    return settings.model_copy(update={"openai_api_key": key})


def test_voice_needs_an_openai_key_and_can_be_switched_off(settings):
    assert voice_available(settings)
    assert not voice_available(without_key(settings))
    assert not voice_available(without_key(settings, SecretStr("   ")))
    off = settings.model_copy(
        update={"voice": settings.voice.model_copy(update={"enabled": False})}
    )
    assert not voice_available(off)


def test_transcribe_sends_the_recording_with_the_finance_vocabulary(settings):
    fake, transcriptions = client(text="  What is a Roth IRA?  ")
    assert transcribe(b"RIFF....", settings, filename="q.wav", client=fake) == "What is a Roth IRA?"
    [call] = transcriptions.calls
    assert call["model"] == "whisper-1" and call["language"] == "en"
    assert call["prompt"] == VOCABULARY and "Roth IRA" in VOCABULARY
    name, audio = call["file"]
    assert name == "q.wav" and audio.read() == b"RIFF...."


def test_transcribe_explains_every_failure(settings, caplog):
    with pytest.raises(VoiceError, match="needs an OpenAI API key"):
        transcribe(b"x", without_key(settings))
    with pytest.raises(VoiceError, match="recording was empty"):
        transcribe(b"", settings, client=client()[0])
    with pytest.raises(VoiceError, match="No speech was recognized"):
        transcribe(b"x", settings, client=client(text="   ")[0])
    caplog.set_level(logging.WARNING)
    with pytest.raises(VoiceError, match="Couldn't transcribe the recording just now"):
        transcribe(b"x", settings, client=client(error=TimeoutError("slow"))[0])
    assert "Transcription failed: TimeoutError" in caplog.text


def test_transcripts_and_recordings_stay_out_of_logs(settings, caplog):
    caplog.set_level(logging.DEBUG)
    transcribe(b"secret audio", settings, client=client(text="my private question")[0])
    assert "my private question" not in caplog.text and "secret audio" not in caplog.text
    assert "Transcribed a recording" in caplog.text


def test_default_client_uses_the_openai_key(settings, monkeypatch):
    import openai

    made = {}

    class FakeOpenAI:
        def __init__(self, api_key, timeout):
            made.update(api_key=api_key, timeout=timeout)
            self.audio = client(text="Hello")[0].audio

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    assert transcribe(b"x", settings) == "Hello"
    assert made == {"api_key": "sk-test-voice-key", "timeout": 30}


def test_speakable_text_leaves_out_the_disclaimer_citations_and_markdown():
    answer = (
        "### What an ETF is\n"
        "An **ETF** holds many stocks [1][2]. See [Investor.gov](https://www.investor.gov).\n"
        "- Low fees\n"
        "1. Trades like a stock [N1]\n"
        "> Costs \\$5 a year\n\n"
        f"{SHORT_DISCLAIMER}"
    )
    assert speakable_text(answer) == (
        "What an ETF is. An ETF holds many stocks. See Investor.gov. Low fees. "
        "Trades like a stock. Costs $5 a year."
    )
    assert speakable_text(SHORT_DISCLAIMER) == ""


def test_settings_secret_is_used_not_printed(settings):
    assert isinstance(settings.openai_api_key, SecretStr)
    assert "sk-test-voice-key" not in repr(settings)
