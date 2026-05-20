from abc import abstractmethod
from typing import AsyncIterable

from livekit import rtc
from livekit.agents import ModelSettings, tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.tts import ChunkedStream


class BaseTTS(tts.TTS):

    @abstractmethod
    async def generator(
            self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
       ...

    def synthesize(
            self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        pass