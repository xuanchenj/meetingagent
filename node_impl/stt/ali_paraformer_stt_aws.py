import asyncio
import json
import logging
import os
import uuid
from typing import AsyncIterable, Optional

import dashscope
import websockets
from livekit import rtc
from livekit.agents import ModelSettings, Agent, NotGivenOr, NOT_GIVEN, APIConnectOptions

from livekit.agents.stt import stt, SpeechEventType, SpeechData, SpeechEvent
from livekit.agents.utils import AudioBuffer

from node_impl.stt.base_stt import BaseSTT

logging.getLogger("dashscope").setLevel(logging.INFO)

dashscope.api_key = os.environ.get("BAILIAN_API_KEY")

logger = logging.getLogger(__name__)

async def send_audio(audio: AsyncIterable[rtc.AudioFrame], websocket, task_id: str):
    async for frame in audio:
        try:
            await websocket.send(bytes(frame.data))
        except:
            pass
    await send_stop(task_id, websocket)

async def send_stop(task_id: str, websocket):
    finsh_cmd = {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex"
        },
        "payload": {
            "input": {}
        }
    }
    await websocket.send(json.dumps(finsh_cmd))
    logger.info(f"[AliParaformerSTTAws] send_stop 结束")

async def start_task(websocket, sample_rate):
    task_id = str(uuid.uuid4())
    run_task_cmd = {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex"
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": "paraformer-realtime-v2",
            "parameters": {
                "format": "pcm", # 音频格式
                "sample_rate": sample_rate, # 采样率
                "disfluency_removal_enabled": False, # 过滤语气词
                "heartbeat":True, # 心跳
                "semantic_punctuation_enabled": True, # 语义标点
                "language_hints": [
                    "zh"
                ] # 指定语言，仅支持paraformer-realtime-v2模型
            },
            "resources": [ #不使用热词功能时，不要传递resources参数
                {
                    "resource_id": "", # paraformer-realtime-v1支持的热词ID
                    "resource_type": "asr_phrase"
                }
            ],
            "input": {}
        }
    }
    await websocket.send(json.dumps(run_task_cmd))


class AliParaformerSTTAws(BaseSTT):
    SERVER_URI = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    API_KEY = os.environ.get("BAILIAN_API_KEY")
    websocket = None
    task_id = None

    def __init__(self, audio_sample_rate: Optional[int] = None):
        self.sample_rate = audio_sample_rate

    async def _recognize_impl(
            self,
            buffer: AudioBuffer,
            *,
            language: NotGivenOr[str] = NOT_GIVEN,
            conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        pass


    async def generator(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        logger.info("enter AliParaformerSTTAws.genertor called")
        self.websocket = await websockets.connect(
            self.SERVER_URI,
            additional_headers={
                "Authorization": f"bearer {self.API_KEY}",
                "X-DashScope-DataInspection": "enable"
            }
        )
        self.task_id = await start_task(self.websocket, self.sample_rate)
        try:
            while True:
                response = await self.websocket.recv()
                if isinstance(response, str):
                    msg_json = json.loads(response)
                    if "header" in msg_json:
                        header = msg_json["header"]

                    if "event" in header:
                        event = header["event"]
                        if event == "task-started":
                            asyncio.get_event_loop().create_task(send_audio(audio=audio, websocket=self.websocket, task_id=self.task_id))
                        elif event == "task-finished":
                            logger.info("stt finished")
                            try:
                                await self.websocket.close()
                            except:
                                pass
                            break

                        elif event == "task-failed":
                            error_msg = msg_json.get("error_message", "未知错误")
                            logger.info(f"stt error {error_msg}")
                            logger.info(f"stt error, msg_json: {msg_json}")
                            self.websocket = await websockets.connect(
                                self.SERVER_URI,
                                additional_headers={
                                    "Authorization": f"bearer {self.API_KEY}",
                                    "X-DashScope-DataInspection": "enable"
                                }
                            )
                            self.task_id = await start_task(self.websocket, self.sample_rate)

                        elif event == "result-generated":
                            sentence = msg_json["payload"]["output"]["sentence"]
                            if "text" in sentence:
                                asr_text = sentence["text"]
                                print("收到asr文本：", asr_text)
                                type = SpeechEventType.INTERIM_TRANSCRIPT
                                if sentence.get("end_time"):
                                    type = SpeechEventType.FINAL_TRANSCRIPT
                                start_time = sentence.get("start_time", 0)
                                end_time = sentence.get("end_time", 0)
                                sd = SpeechData(
                                    language = "zh",
                                    text= asr_text,
                                    start_time = start_time,
                                    end_time = end_time,
                                )
                                yield stt.SpeechEvent(type=type, alternatives=[sd])
                else:
                    continue
        finally:
            try:
                await self.websocket.close()
            except:
                pass
