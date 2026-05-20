from abc import abstractmethod
from typing import AsyncIterable, Optional

from livekit import rtc
from livekit.agents import ModelSettings, stt


class BaseSTT(stt.STT):

    @abstractmethod
    async def generator(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        ...