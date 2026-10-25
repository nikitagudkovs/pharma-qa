"""One LLM provider behind a small interface. Every call returns a Pydantic
model or typed tool calls, never free text to be parsed.
"""
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output)
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        p_in, p_out = PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1_000_000


@dataclass
class ToolCall:
    name: str
    input: dict


@dataclass
class CallLog:
    """Every LLM call made while answering one question, for the trace."""
    calls: list[dict] = field(default_factory=list)

    def add(self, step: str, usage: Usage, latency_ms: int) -> None:
        self.calls.append({"step": step, "model": usage.model, "input_tokens": usage.input_tokens,
                           "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd,
                           "latency_ms": latency_ms})

    @property
    def cost_usd(self) -> float:
        return sum(c["cost_usd"] for c in self.calls)


class LLMClient(Protocol):
    def parse(self, model: str, system: str, user: str, output: type[T], effort: str = "low") -> tuple[T, Usage]: ...

    def tool_calls(self, model: str, system: str, user: str, tools: list[dict],
                   effort: str = "low") -> tuple[list[ToolCall], Usage]: ...


def _thinking(model: str) -> dict:
    # Haiku 5.5 may run without thinking at effort <= high; Sonnet 5.5 can't
    # disable it, so it runs adaptive at low effort.
    return {"type": "disabled"} if model.startswith("claude-haiku") else {"type": "adaptive"}


class AnthropicClient:
    def __init__(self):
        self.client = anthropic.Anthropic(max_retries=4, timeout=60)

    def parse(self, model, system, user, output, effort="low"):
        resp = self.client.messages.parse(
            model=model,
            max_tokens=4096,
            system=system,
            thinking=_thinking(model),
            output_config={"effort": effort},
            messages=[{"role": "user", "content": user}],
            output_format=output,
        )
        if resp.stop_reason == "refusal":
            raise LLMRefusal(getattr(resp.stop_details, "category", None))
        return resp.parsed_output, Usage(model, resp.usage.input_tokens, resp.usage.output_tokens)

    def tool_calls(self, model, system, user, tools, effort="low"):
        resp = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            thinking=_thinking(model),
            output_config={"effort": effort},
            tools=tools,
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise LLMRefusal(getattr(resp.stop_details, "category", None))
        calls = [ToolCall(b.name, b.input) for b in resp.content if b.type == "tool_use"]
        return calls, Usage(model, resp.usage.input_tokens, resp.usage.output_tokens)


class LLMRefusal(RuntimeError):
    pass
