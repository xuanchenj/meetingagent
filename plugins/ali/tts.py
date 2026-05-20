from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass

from websockets.asyncio.client import connect as ws_connect

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger

SERVER_URI = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
NUM_CHANNELS = 1

# 句子硬断句符
_SENTENCE_ENDS = frozenset("。！？!?\n")
# 软断句符（句子过长时在此处切分）
_SOFT_ENDS = frozenset("，,；;、")
# 超过此字数未遇到任何断句符则强制切分
_MAX_CHARS_BEFORE_SPLIT = 60


@dataclass
class _TTSOptions:
    api_key: str
    model: str
    voice: str
    sample_rate: int
    volume: int
    rate: float
    pitch: float


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
        """
        创建阿里云 CosyVoice TTS 实例。

        Args:
            api_key: 阿里云 API Key，未提供时从 BAILIAN_API_KEY 环境变量读取。
            model: TTS 模型名称，默认 cosyvoice-v3-flash。
            voice: 音色 ID，默认 longanhuan。
            sample_rate: 音频采样率（Hz），默认 16000。
            volume: 音量（0-100），默认 50。
            rate: 语速倍率（0.5-2.0），默认 1.0。
            pitch: 音调（0.5-2.0），默认 1.0。
        """
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
        return SynthesizeStream(tts=self, conn_options=conn_options)


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
        try:
            async with ws_connect(
                SERVER_URI,
                additional_headers={
                    "Authorization": f"bearer {self._opts.api_key}",
                    "X-DashScope-DataInspection": "enable",
                },
            ) as ws:
                await ws.send(json.dumps(_build_run_task_cmd(task_id, self._opts)))
                while True:
                    response = await ws.recv()
                    if isinstance(response, bytes):
                        output_emitter.push(response)
                        continue
                    msg = json.loads(response)
                    event = msg.get("header", {}).get("event", "")
                    if event == "task-started":
                        await ws.send(json.dumps({
                            "header": {"action": "continue-task", "task_id": task_id, "streaming": "duplex"},
                            "payload": {"input": {"text": self._input_text}},
                        }))
                        await ws.send(json.dumps({
                            "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                            "payload": {"input": {}},
                        }))
                    elif event == "task-finished":
                        output_emitter.flush()
                        break
                    elif event == "task-failed":
                        error_msg = msg.get("header", {}).get("error_message", "未知错误")
                        raise APIConnectionError(f"阿里 TTS 任务失败: {error_msg}")
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """
    流式文本合成，仿照 Deepgram 的 segment 机制：

    1. 将 LLM 流式文本按句子边界切分
    2. 每个句子 = 一个独立的 Ali TTS task = 一个 emitter segment
    3. 逐句顺序合成，等 task-finished 后再开始下一句
    4. 每句各自建立 WebSocket 连接，用完即断

    效果：任何时刻 pipeline 里最多只有「当前句子」的音频，
    打断后的尾音从「整个回复」缩减到「最多一句话」。
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = tts._opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        sentence_ch: utils.aio.Chan[str] = utils.aio.Chan()

        async def split_sentences() -> None:
            """将 LLM token 流按句子边界切分，推入 sentence_ch。"""
            buf = ""
            async for data in self._input_ch:
                if isinstance(data, str):
                    buf += data
                    while True:
                        pos = next(
                            (i for i, c in enumerate(buf) if c in _SENTENCE_ENDS), -1
                        )
                        if pos == -1:
                            break
                        sentence = buf[: pos + 1].strip()
                        buf = buf[pos + 1 :]
                        if sentence:
                            sentence_ch.send_nowait(sentence)
                    # 超长时按软断句符或强制切分
                    while len(buf) >= _MAX_CHARS_BEFORE_SPLIT:
                        soft_pos = next(
                            (i for i, c in enumerate(buf) if c in _SOFT_ENDS), -1
                        )
                        split_at = soft_pos if soft_pos > 0 else _MAX_CHARS_BEFORE_SPLIT - 1
                        sentence = buf[: split_at + 1].strip()
                        buf = buf[split_at + 1 :]
                        if sentence:
                            sentence_ch.send_nowait(sentence)
                elif isinstance(data, self._FlushSentinel):
                    break
            if buf.strip():
                sentence_ch.send_nowait(buf.strip())
            sentence_ch.close()

        async def synthesize_sentences() -> None:
            """逐句调用 Ali TTS，等 task-finished 后再开始下一句。"""
            async for sentence in sentence_ch:
                await _run_ws_sentence(
                    sentence=sentence,
                    output_emitter=output_emitter,
                    opts=self._opts,
                    conn_options=self._conn_options,
                    mark_started_cb=self._mark_started,
                )

        tasks = [
            asyncio.create_task(split_sentences()),
            asyncio.create_task(synthesize_sentences()),
        ]
        try:
            await asyncio.gather(*tasks)
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)


async def _run_ws_sentence(
    *,
    sentence: str,
    output_emitter: tts.AudioEmitter,
    opts: _TTSOptions,
    conn_options: APIConnectOptions,
    mark_started_cb,
) -> None:
    """
    对应 Deepgram 的 _run_ws()：为单个句子完整执行一次 Ali TTS task，
    收到 task-finished 后调用 end_segment() 并返回。
    """
    task_id = str(uuid.uuid4())
    segment_id = utils.shortuuid()
    output_emitter.start_segment(segment_id=segment_id)

    task_started = asyncio.Event()

    try:
        async with ws_connect(
            SERVER_URI,
            additional_headers={
                "Authorization": f"bearer {opts.api_key}",
                "X-DashScope-DataInspection": "enable",
            },
        ) as ws:
            await ws.send(json.dumps(_build_run_task_cmd(task_id, opts)))

            async def send_task() -> None:
                await task_started.wait()
                mark_started_cb()
                await ws.send(json.dumps({
                    "header": {"action": "continue-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {"input": {"text": sentence}},
                }))
                await ws.send(json.dumps({
                    "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {"input": {}},
                }))
                logger.debug("[AliTTS] 句子已发送: %s", sentence[:20])

            async def recv_task() -> None:
                while True:
                    response = await ws.recv()
                    if isinstance(response, bytes):
                        output_emitter.push(response)
                        continue
                    msg = json.loads(response)
                    event = msg.get("header", {}).get("event", "")
                    if event == "task-started":
                        logger.debug("[AliTTS] task-started (%.10s…)", sentence)
                        task_started.set()
                    elif event == "task-finished":
                        logger.debug("[AliTTS] task-finished (%.10s…)", sentence)
                        output_emitter.end_segment()
                        break
                    elif event == "task-failed":
                        error_msg = msg.get("header", {}).get("error_message", "未知错误")
                        logger.error("[AliTTS] task-failed: %s", error_msg)
                        raise APIConnectionError(f"阿里 TTS 任务失败: {error_msg}")

            inner_tasks = [
                asyncio.create_task(send_task()),
                asyncio.create_task(recv_task()),
            ]
            try:
                await asyncio.gather(*inner_tasks)
            finally:
                task_started.set()
                await utils.aio.gracefully_cancel(*inner_tasks)

    except APIConnectionError:
        raise
    except Exception as e:
        raise APIConnectionError() from e


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
