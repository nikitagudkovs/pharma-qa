from app.llm import ToolCall, Usage


class FakeLLM:
    """Scripted LLMClient: returns canned outputs per output model / tool step."""

    def __init__(self, parsed: dict | None = None, tool_calls: list[ToolCall] | None = None):
        self.parsed = parsed or {}
        self._tool_calls = tool_calls or []
        self.calls: list[str] = []

    def parse(self, model, system, user, output, effort="low"):
        self.calls.append(output.__name__)
        return output.model_validate(self.parsed[output.__name__]), Usage(model, 100, 20)

    def tool_calls(self, model, system, user, tools, effort="low"):
        self.calls.append("tool_calls")
        return self._tool_calls, Usage(model, 300, 50)
