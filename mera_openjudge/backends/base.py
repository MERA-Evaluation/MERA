from abc import ABC, abstractmethod
from typing import Sequence


class BaseJudgeBackend(ABC):
    @abstractmethod
    def generate(self, prompts: Sequence[str]):
        raise NotImplementedError
