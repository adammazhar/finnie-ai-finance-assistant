"""Voice for the chat: speech to text for questions, and answer text prepared for reading aloud.

- **Speech to text** uses OpenAI's transcription API (``voice.model`` in config.yaml,
  ``whisper-1``), with the same ``OPENAI_API_KEY`` as the chat. A short vocabulary
  prompt helps it spell finance terms ("ETF", "401(k)", "Roth IRA"). The recording is sent
  once for transcription; Finnie doesn't store it, and neither the audio nor the transcript
  is logged.
- **Reading aloud** happens in the browser with its built-in speech synthesis, so it needs no
  key. :func:`speakable_text` turns an answer into plain sentences for it: no disclaimer,
  citation markers, or markdown.
"""

from __future__ import annotations

import io
import logging
import re
import time
from typing import Any

from src.core.config import Settings
from src.core.guardrails import strip_disclaimer

logger = logging.getLogger(__name__)

VOCABULARY = (
    "Finnie, ETF, index fund, mutual fund, 401(k), 403(b), IRA, Roth IRA, HSA, 529, RMD, "
    "S&P 500, Nasdaq, Dow Jones, expense ratio, dividend, Vanguard, VTI, VOO, SPY, bonds, "
    "capital gains, Monte Carlo"
)


class VoiceError(RuntimeError):
    """Transcription isn't available or failed; the message is safe to show the user."""


def voice_available(settings: Settings) -> bool:
    """Whether voice input can be offered: it's enabled and an OpenAI key is set."""
    key = settings.openai_api_key
    return settings.voice.enabled and key is not None and bool(key.get_secret_value().strip())


def transcribe(
    audio: bytes, settings: Settings, *, filename: str = "question.wav", client: Any = None
) -> str:
    """The text of a recorded question. Raises :class:`VoiceError` with a plain message when
    voice isn't set up, the recording is empty, or the transcription service fails."""
    if not voice_available(settings):
        raise VoiceError("Voice input needs an OpenAI API key (OPENAI_API_KEY in .env).")
    if not audio:
        raise VoiceError("The recording was empty. Please try again.")
    if client is None:
        from openai import OpenAI

        key = settings.openai_api_key
        assert key is not None  # checked by voice_available
        client = OpenAI(api_key=key.get_secret_value(), timeout=settings.voice.timeout_s)
    started = time.perf_counter()
    try:
        result = client.audio.transcriptions.create(
            model=settings.voice.model,
            file=(filename, io.BytesIO(audio)),
            language=settings.voice.language,
            prompt=VOCABULARY,
        )
    except Exception as exc:
        logger.warning("Transcription failed: %s", type(exc).__name__)
        raise VoiceError(
            "Couldn't transcribe the recording just now. Please type instead."
        ) from None
    text = str(getattr(result, "text", "") or "").strip()
    logger.info(
        "Transcribed a recording",
        extra={"seconds": round(time.perf_counter() - started, 2), "chars": len(text)},
    )
    if not text:
        raise VoiceError("No speech was recognized. Please try again, a little closer to the mic.")
    return text


_CITATION = re.compile(r"\[(?:N)?\d{1,2}\]")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKUP = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s?)|[*_`~]|\\(?=[$*_])")


def speakable_text(answer: str) -> str:
    """An answer as plain text for speech: without the disclaimer, citation markers like
    ``[2]``, links, or markdown, and with list items and paragraphs as separate sentences."""
    text = strip_disclaimer(answer)
    text = _LINK.sub(r"\1", text)
    text = _CITATION.sub("", text)
    text = _MARKUP.sub("", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [line if line[-1] in ".!?:" else f"{line}." for line in lines]
    return re.sub(r"\s+([.,;:!?])", r"\1", " ".join(sentences)).strip()
