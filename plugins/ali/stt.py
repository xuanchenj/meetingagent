from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from typing import AsyncIterable, Optional

import websockets
from livekit.agents.stt import SpeechEventType, SpeechData
from websockets.asyncio.client import ClientConnection, connect as ws_connect
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    stt,
    utils, ModelSettings,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from .log import logger

SERVER_URI = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

# 每次向阿里 ASR 发送的音频块时长（毫秒），避免过短片段影响识别
_AUDIO_CHUNK_MS = 200


@dataclass
class _STTOptions:
    api_key: str
    sample_rate: int
    model: str
    language: str

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


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        sample_rate: int = 16000,
        model: str = "paraformer-realtime-v2",
        language: str = "zh",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )
        resolved_api_key = api_key or os.environ.get("BAILIAN_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "阿里云 API Key 未设置，请传入 api_key 参数或设置 BAILIAN_API_KEY 环境变量"
            )

        self._opts = _STTOptions(
            api_key=resolved_api_key,
            sample_rate=sample_rate,
            model=model,
            language=language,

        )

    async def generator(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        logger.info("enter AliParaformerSTTAws.genertor called")
        self.websocket = await websockets.connect(
            SERVER_URI,
            additional_headers={
                "Authorization": f"bearer {self._opts.api_key}",
                "X-DashScope-DataInspection": "enable"
            }
        )
        self.task_id = await start_task(self.websocket, self._opts.sample_rate)
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
                                SERVER_URI,
                                additional_headers={
                                    "Authorization": f"bearer {self._opts.api_key}",
                                    "X-DashScope-DataInspection": "enable"
                                }
                            )
                            self.task_id = await start_task(self.websocket, self._opts.sample_rate)

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

    @property
    def provider(self) -> str:
        return "Ali-Paraformer"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("阿里 Paraformer 不支持非流式识别")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(stt=self, conn_options=conn_options, opts=self._opts)


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        opts: _STTOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._opts = opts

    async def _run(self) -> None:
        try:
            async with ws_connect(
                SERVER_URI,
                additional_headers={
                    "Authorization": f"bearer {self._opts.api_key}",
                    "X-DashScope-DataInspection": "enable",
                },
            ) as ws:
                await self._run_ws(ws)
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_ws(self, ws: ClientConnection) -> None:
        task_id = str(uuid.uuid4())
        task_started = asyncio.Event()

        async def send_task() -> None:
            run_task_cmd = {
                "header": {
                    "action": "run-task",
                    "task_id": task_id,
                    "streaming": "duplex",
                },
                "payload": {
                    "task_group": "audio",
                    "task": "asr",
                    "function": "recognition",
                    "model": self._opts.model,
                    "parameters": {
                        "format": "pcm",
                        "sample_rate": self._opts.sample_rate,
                        "disfluency_removal_enabled": False,
                        "heartbeat": True,
                        "semantic_punctuation_enabled": True,
                        "language_hints": [self._opts.language],
                    },
                    "resources": [],
                    "input": {},
                },
            }
            await ws.send(json.dumps(run_task_cmd))

            await task_started.wait()

            samples_per_chunk = self._opts.sample_rate * _AUDIO_CHUNK_MS // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    for chunk in audio_bstream.flush():
                        await ws.send(bytes(chunk.data))
                else:
                    for chunk in audio_bstream.write(data.data.tobytes()):
                        await ws.send(bytes(chunk.data))

            for chunk in audio_bstream.flush():
                await ws.send(bytes(chunk.data))

            finish_cmd = {
                "header": {
                    "action": "finish-task",
                    "task_id": task_id,
                    "streaming": "duplex",
                },
                "payload": {"input": {}},
            }
            await ws.send(json.dumps(finish_cmd))
            logger.debug("[AliSTT] 已发送 finish-task 指令")

        async def recv_task() -> None:
            while True:
                response = await ws.recv()
                if not isinstance(response, str):
                    continue

                msg = json.loads(response)
                header = msg.get("header", {})
                event = header.get("event", "")

                if event == "task-started":
                    logger.debug("[AliSTT] task-started 收到")
                    task_started.set()

                elif event == "result-generated":
                    sentence = msg.get("payload", {}).get("output", {}).get("sentence", {})
                    text = sentence.get("text", "")
                    if not text:
                        continue

                    is_final = bool(sentence.get("end_time"))
                    event_type = (
                        stt.SpeechEventType.FINAL_TRANSCRIPT
                        if is_final
                        else stt.SpeechEventType.INTERIM_TRANSCRIPT
                    )
                    start_time = sentence.get("start_time", 0) / 1000.0
                    end_time = sentence.get("end_time", 0) / 1000.0 if is_final else 0.0

                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=event_type,
                            alternatives=[
                                stt.SpeechData(
                                    language=self._opts.language,
                                    text=text,
                                    start_time=start_time,
                                    end_time=end_time,
                                )
                            ],
                        )
                    )
                    logger.debug(
                        "[AliSTT] %s: %s",
                        "FINAL" if is_final else "INTERIM",
                        text,
                    )

                elif event == "task-finished":
                    logger.debug("[AliSTT] task-finished")
                    break

                elif event == "task-failed":
                    error_msg = msg.get("header", {}).get("error_message", "未知错误")
                    logger.error("[AliSTT] task-failed: %s", error_msg)
                    raise APIConnectionError(f"阿里 STT 任务失败: {error_msg}")

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
