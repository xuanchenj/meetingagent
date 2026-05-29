import logging
import os
from typing import AsyncIterable, Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, RoomInputOptions, llm, \
    SessionUsageUpdatedEvent, stt, ModelSettings
from livekit.plugins import openai, silero

from plugins.ali import STT as AliSTT, TTS as AliTTS

load_dotenv()

server = AgentServer()

logger = logging.getLogger(__name__)


def prewarm(proc: JobProcess):
    logger.info("进入预热函数")
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


class MeetingAgent(Agent):
    def __init__(
        self,
        instructions: str | None,
        tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
        ttsn:AliTTS = None,
        sttn:AliSTT = None
    ) -> None:
        self.sttn = sttn
        self.ttsn = ttsn
        super().__init__(instructions=instructions, tools=tools)

    async def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        async for itme in self.sttn.generator(audio, model_settings):
            yield itme

    async def tts_node(
            self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        async for itme in self.ttsn.generator(text, model_settings):
            yield itme

@server.rtc_session(agent_name="meeting_agent")
async def entry_point(ctx: JobContext):
    logger.info("enter entry point!!!!!!!!!!!!")

    stt = AliSTT(sample_rate=16000)
    logger.info("stt init")

    tts = AliTTS(voice="longanhuan", sample_rate=16000)
    logger.info("tts init")

    llm_model = openai.LLM(
        model="qwen3.6-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("BAILIAN_API_KEY"),
        temperature=0.7,
        extra_body={"enable_thinking":False}
    )
    logger.info("llm init")

    agent = MeetingAgent(
        instructions=(
            "你是一个情感陪聊助手，你叫欢欢。"
            "请你以欢快的语气（禁止输出动作申请描述信息，因为你的输出将作为tts的文本播报给用户）陪用户聊天"
        ),
        ttsn=tts,
        sttn=stt
    )

    await ctx.connect()

    session = AgentSession(
        stt=stt,
        tts=tts,
        llm=llm_model,
        vad=ctx.proc.userdata.get("vad"),
        turn_detection="vad",
        min_interruption_words=1,
        min_interruption_duration=0.3,
        user_away_timeout=10,
    )

    @session.on("session_usage_updated")
    def on_session_usage_updated(ev: SessionUsageUpdatedEvent):
        for usage in ev.usage.model_usage:
            logger.info(f"usage log：{usage.provider}/{usage.model}: {usage}")

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(audio_sample_rate=16000),
    )

    await session.say("你好啊，我是欢欢，你今天心情怎么样啊？", allow_interruptions=False)
