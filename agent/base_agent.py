from livekit.agents import Agent, llm


class BaseAgent(Agent):
    def __init__(
        self,
        instructions: str,
        tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
    ) -> None:
        super().__init__(instructions=instructions, tools=tools)
