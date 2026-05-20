import asyncio
from livekit.agents import ChatContext, ChatMessage
from livekit.plugins import openai

llm_model = openai.LLM(
    model="qwen3.6-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-94072328cdfd4e2bb063f450c7564387",
    temperature=0.7,
    extra_body={"enable_thinking": False},
)

async def main():
    ctx = ChatContext([
        ChatMessage(role="user", content=["你好"])
    ])
    stream = llm_model.chat(chat_ctx=ctx)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
