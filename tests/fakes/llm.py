"""A scripted chat model for tests. No network, fully deterministic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field


class FakeChatModel(BaseChatModel):
    """Returns scripted replies in order and records every call.

    - ``responses``: ``str``, ``AIMessage`` (may carry ``tool_calls``), or an ``Exception``
      to raise. The last response repeats once the script is exhausted.
    - ``structured_responses``: objects returned by ``with_structured_output`` runnables,
      in order across all of them (one script per model, like ``responses``); an
      ``Exception`` entry is raised.
    """

    responses: list[Any] = Field(default_factory=lambda: ["ok"])
    structured_responses: list[Any] = Field(default_factory=list)
    calls: list[list[Any]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    structured_schemas: list[Any] = Field(default_factory=list)
    structured_calls: int = 0
    generate_calls: int = 0
    name: str = "fake"

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _next(self, script: list[Any], index: int) -> Any:
        if not script:
            raise AssertionError("FakeChatModel script is empty")
        item = script[min(index, len(script) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        item = self._next(self.responses, self.generate_calls)
        self.generate_calls += 1
        message = item if isinstance(item, AIMessage) else AIMessage(content=str(item))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeChatModel:  # type: ignore[override]
        self.bound_tools = list(tools)
        return self

    def with_structured_output(  # type: ignore[override]
        self, schema: Any, **kwargs: Any
    ) -> Runnable[Any, Any]:
        self.structured_schemas.append(schema)

        def respond(messages: Any) -> Any:
            self.calls.append(messages if isinstance(messages, list) else [messages])
            item = self._next(self.structured_responses, self.structured_calls)
            self.structured_calls += 1
            if (
                isinstance(schema, type)
                and issubclass(schema, BaseModel)
                and isinstance(item, dict)
            ):
                return schema.model_validate(item)
            return item

        return RunnableLambda(respond)
