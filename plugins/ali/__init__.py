"""阿里云语音插件 for LiveKit Agents（Paraformer STT + CosyVoice TTS）"""

from .stt import STT, SpeechStream
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "TTS",
    "ChunkedStream",
    "SynthesizeStream",
    "__version__",
]

from livekit.agents import Plugin
from .log import logger


class AliPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(AliPlugin())
