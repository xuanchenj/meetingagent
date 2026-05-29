from __future__ import annotations

import asyncio
import json
import os
import uuid
import weakref
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, AsyncIterable

import websockets
from livekit import rtc
from websockets.asyncio.client import ClientConnection, connect as ws_connect

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils, ModelSettings,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger

SERVER_URI = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    api_key: str
    model: str
    voice: str
    sample_rate: int
    volume: int
    rate: float
    pitch: float


def _build_run_task_cmd(task_id: str, opts: _TTSOptions) -> dict:
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": opts.model,
            "parameters": {
                "text_type": "PlainText",
                "voice": opts.voice,
                "format": "pcm",
                "sample_rate": opts.sample_rate,
                "volume": opts.volume,
                "rate": opts.rate,
                "pitch": opts.pitch,
            },
            "input": {},
        },
    }


def _build_continue_task_cmd(task_id: str, text: str) -> dict:
    return {
        "header": {
            "action": "continue-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {"text": text}},
    }


def _build_finish_task_cmd(task_id: str) -> dict:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


async def _cancel_tasks(tasks: list[asyncio.Task]) -> None:
    """取消并等待后台任务结束，避免 Task exception was never retrieved。"""
    if not tasks:
        return
    for t in tasks:
        if not t.done():
            t.cancel()
    await utils.aio.gracefully_cancel(*tasks)


async def _close_ws_quietly(ws: ClientConnection | None) -> None:
    if ws is None:
        return
    try:
        await ws.close()
    except Exception:
        pass

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


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "cosyvoice-v3-flash",
        voice: str = "longanhuan",
        sample_rate: int = 16000,
        volume: int = 50,
        rate: float = 1.0,
        pitch: float = 1.0,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        resolved_api_key = api_key or os.environ.get("BAILIAN_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "阿里云 API Key 未设置，请传入 api_key 参数或设置 BAILIAN_API_KEY 环境变量"
            )

        self._opts = _TTSOptions(
            api_key=resolved_api_key,
            model=model,
            voice=voice,
            sample_rate=sample_rate,
            volume=max(0, min(100, volume)),
            rate=max(0.5, min(2.0, rate)),
            pitch=max(0.5, min(2.0, pitch)),
        )
        self._streams: weakref.WeakSet[SynthesizeStream] = weakref.WeakSet()


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
                "model": "cosyvoice-v3-flash",
                "parameters": {
                    "text_type": "PlainText",
                    "voice": self._opts.voice,           # 使用实例的音色ID
                    "format": "pcm",
                    "sample_rate": self._opts.sample_rate,
                    "volume": self._opts.volume,            # 使用实例的音量
                    "rate": self._opts.rate,                # 使用实例的语速倍率
                    "pitch": self._opts.pitch               # 使用实例的音调
                },
                "input": {}
            }
        }
        async with ws_connect(
                SERVER_URI,
                additional_headers={
                    "Authorization": f"bearer {self._opts.api_key}",
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
                                asyncio.get_event_loop().create_task(send_text(taskid, websocket, text))

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

    @property
    def provider(self) -> str:
        return "Ali-CosyVoice"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class _WsSession:
    """单次 Ali TTS WebSocket 会话：统一管理 send/recv 任务与连接回收。"""

    def __init__(
        self,
        *,
        ws: ClientConnection,
        task_id: str,
        opts: _TTSOptions,
    ) -> None:
        self._ws = ws
        self._task_id = task_id
        self._opts = opts
        self._tasks: list[asyncio.Task] = []
        self._task_started = asyncio.Event()
        self._finish_sent = False
        self._closed = False

    @property
    def task_started(self) -> asyncio.Event:
        return self._task_started

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task

    async def run_send(
        self,
        input_ch,
        *,
        flush_sentinel: Any,
        mark_started_cb=None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        """从 input 通道读取文本并发送 continue-task，结束时发一次 finish-task。"""
        try:
            await self._task_started.wait()

            async for data in input_ch:
                if isinstance(data, str):
                    if not data:
                        continue
                    if mark_started_cb:
                        mark_started_cb()
                    await self._ws.send(
                        json.dumps(_build_continue_task_cmd(self._task_id, data))
                    )
                    if on_text:
                        on_text(data)
                elif isinstance(data, flush_sentinel):
                    break

            await self._send_finish()
        except asyncio.CancelledError:
            raise
        except Exception:
            # 发送失败时仍尝试 finish，便于服务端释放 task
            await self._send_finish()
            raise

    async def run_recv(
        self,
        *,
        on_audio: Callable[[bytes], None],
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        """接收服务端事件与音频。"""
        try:
            while True:
                response = await self._ws.recv()

                if isinstance(response, bytes):
                    on_audio(response)
                    continue

                msg = json.loads(response)
                event = msg.get("header", {}).get("event", "")

                if event == "task-started":
                    logger.debug("[AliTTS] task-started")
                    self._task_started.set()
                    if on_started:
                        on_started()
                elif event == "task-finished":
                    logger.debug("[AliTTS] task-finished")
                    if on_finished:
                        on_finished()
                    break
                elif event == "task-failed":
                    error_msg = msg.get("header", {}).get("error_message", "未知错误")
                    logger.error("[AliTTS] task-failed: %s", error_msg)
                    raise APIConnectionError(f"阿里 TTS 任务失败: {error_msg}")
        except asyncio.CancelledError:
            raise

    async def _send_finish(self) -> None:
        if self._finish_sent or self._closed:
            return
        if not self._task_started.is_set():
            return
        self._finish_sent = True
        try:
            await self._ws.send(json.dumps(_build_finish_task_cmd(self._task_id)))
            logger.debug("[AliTTS] finish-task 已发送")
        except Exception as e:
            logger.debug("[AliTTS] finish-task 发送失败: %s", e)

    async def shutdown(self, *, cancel_tasks: bool = True) -> None:
        """回收：取消后台任务、尝试 finish-task、关闭连接。"""
        if self._closed:
            return
        self._closed = True
        self._task_started.set()

        if cancel_tasks:
            await _cancel_tasks(self._tasks)
        self._tasks.clear()

        await self._send_finish()
        await _close_ws_quietly(self._ws)


class ChunkedStream(tts.ChunkedStream):
    """非流式合成（synthesize / session.say() 场景）。"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = tts._opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        task_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=task_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        session: _WsSession | None = None
        try:
            async with ws_connect(
                SERVER_URI,
                additional_headers={
                    "Authorization": f"bearer {self._opts.api_key}",
                    "X-DashScope-DataInspection": "enable",
                },
            ) as ws:
                session = _WsSession(ws=ws, task_id=task_id, opts=self._opts)
                await ws.send(json.dumps(_build_run_task_cmd(task_id, self._opts)))

                async def send_once() -> None:
                    await session.task_started.wait()
                    await ws.send(
                        json.dumps(_build_continue_task_cmd(task_id, self._input_text))
                    )
                    await session._send_finish()

                recv_task = session.spawn(
                    session.run_recv(
                        on_audio=output_emitter.push,
                        on_finished=output_emitter.flush,
                    )
                )
                send_task = session.spawn(send_once())
                await asyncio.gather(recv_task, send_task)

        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if session is not None:
                await session.shutdown()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = tts._opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        task_id = str(uuid.uuid4())
        segment_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=task_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=segment_id)

        session: _WsSession | None = None
        try:
            async with ws_connect(
                SERVER_URI,
                additional_headers={
                    "Authorization": f"bearer {self._opts.api_key}",
                    "X-DashScope-DataInspection": "enable",
                },
            ) as ws:
                session = _WsSession(ws=ws, task_id=task_id, opts=self._opts)
                await ws.send(json.dumps(_build_run_task_cmd(task_id, self._opts)))

                recv_task = session.spawn(
                    session.run_recv(
                        on_audio=output_emitter.push,
                        on_finished=output_emitter.end_segment,
                    )
                )
                send_task = session.spawn(
                    session.run_send(
                        self._input_ch,
                        flush_sentinel=self._FlushSentinel,
                        mark_started_cb=self._mark_started,
                        on_text=lambda t: logger.debug("[AliTTS] 发送文本: %s", t),
                    )
                )
                await asyncio.gather(recv_task, send_task)

        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if session is not None:
                await session.shutdown()
