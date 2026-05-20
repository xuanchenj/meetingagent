import asyncio
import json
import logging
import os
import uuid
from typing import AsyncIterable

import websockets
from livekit import rtc
from livekit.agents import ModelSettings

from node_impl.tts.base_tts import BaseTTS

logging.getLogger("dashscope").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

class AliCosyVoiceTTS(BaseTTS):
    """
    阿里云CosyVoice TTS实现
    
    支持从metadata中获取以下参数：
    - voice_id: 音色ID
    - speech_rate: 语速
    - volume: 音量
    - rate: 语速倍率
    - pitch: 音调
    """
    
    #TODO：配置信息暂时先写死，有时间再改成配置形式
    API_KEY = os.environ.get("BAILIAN_API_KEY")
    SERVER_URI = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
    sample_rate = 8000
    
    def __init__(self, voice_id: str, 
                 volume: int = 50, rate: float = 1.0, pitch: float = 1.0):
        """
        初始化阿里云TTS
        
        Args:
            voice_id: 音色ID
            volume: 音量，默认50
            rate: 语速倍率，默认1.0
            pitch: 音调，默认1.0
        """
        logger.info(f"阿里云TTS初始化 - 音色: {voice_id}, 音量: {volume}, 语速倍率: {rate}, 音调: {pitch}")
        self.voice_id = voice_id
        self.volume = volume
        self.rate = rate
        self.pitch = pitch
        
        # 参数验证和范围限制
        self._validate_and_limit_params()
        
        logger.info(f"阿里云TTS初始化 - 音色: {self.voice_id}"
                   f"音量: {self.volume}, 语速倍率: {self.rate}, 音调: {self.pitch}")
    
    def _validate_and_limit_params(self):
        """验证和限制参数范围"""
        if not self.voice_id:
            raise ValueError(f"音色ID不能为空")
        
        # 音量范围限制
        if self.volume < 0:
            self.volume = 0
            raise ValueError(f"音量参数过小，已调整为最小值: 0")
        elif self.volume > 100:
            self.volume = 100
            raise ValueError(f"音量参数过大，已调整为最大值: 100")
        
        # 语速倍率范围限制
        if self.rate < 0.5:
            self.rate = 0.5
            raise ValueError(f"语速倍率参数过小，已调整为最小值: 0.5")
        elif self.rate > 2.0:
            self.rate = 2.0
            raise ValueError(f"语速倍率参数过大，已调整为最大值: 2.0")
        
        # 音调范围限制
        if self.pitch < 0.5:
            self.pitch = 0.5
            raise ValueError(f"音调参数过小，已调整为最小值: 0.5")
        elif self.pitch > 2.0:
            self.pitch = 2.0
            raise ValueError(f"音调参数过大，已调整为最大值: 2.0")

    @staticmethod
    async def send_text(task_id, websocket, text:AsyncIterable[str]):
        try:
            async for t in text:
                cmd = {
                    "header": {
                        "action": "continue-task",
                        "task_id": task_id,
                        "streaming": "duplex"
                    },
                    "payload": {
                        "input": {
                            "text": t
                        }
                    }
                }
                await websocket.send(json.dumps(cmd))
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
        except websockets.exceptions.ConnectionClosed as e:
            logger.info("!!!!producer close connection close!!!!!!")

    #TODO：需要改成wscoket可复用方式，避免每次TTS都创建新链接
    async def generator(
            self, text: AsyncIterable[str],  model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        taskid = str(uuid.uuid4())
        # 构造 run-task 指令
        run_task_cmd = {
            "header": {
                "action": "run-task",
                "task_id": taskid,
                "streaming": "duplex"
            },
            "payload": {
                "task_group": "audio",
                "task": "tts",
                "function": "SpeechSynthesizer",
                "model": "cosyvoice-v3.5-flash",
                "parameters": {
                    "text_type": "PlainText",
                    "voice": self.voice_id,           # 使用实例的音色ID
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                    "volume": self.volume,            # 使用实例的音量
                    "rate": self.rate,                # 使用实例的语速倍率
                    "pitch": self.pitch               # 使用实例的音调
                },
                "input": {}
            }
        }
        async with websockets.connect(
                self.SERVER_URI,
                additional_headers={
                    "Authorization": f"bearer {self.API_KEY}",
                    "X-DashScope-DataInspection": "enable"
                }
        ) as websocket:
            await websocket.send(json.dumps(run_task_cmd))

            while True:
                try:
                    response = await websocket.recv()
                    if isinstance(response, str):
                        msg_json = json.loads(response)
                        if "header" in msg_json:
                            header = msg_json["header"]

                        if "event" in header:
                            event = header["event"]

                            if event == "task-started":
                                asyncio.get_event_loop().create_task(AliCosyVoiceTTS.send_text(taskid, websocket, text))

                            elif event == "task-finished":
                                logger.info("tts finished")
                                break

                            elif event == "task-failed":
                                error_msg = msg_json.get("error_message", "未知错误")
                                logger.info(f"tts error {msg_json}")
                                logger.info(f"tts error {error_msg}")
                                break
                    else:
                        yield rtc.AudioFrame(response, self.sample_rate, 1, 16)
                except websockets.exceptions.ConnectionClosed:
                    logger.exception("tts Connection closed")
                    break
                except asyncio.CancelledError:
                    logger.info("TTS任务被取消")
                    break