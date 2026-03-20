from abc import ABC, abstractmethod

class Collector(ABC):
    @abstractmethod
    def collect(self) -> None:
        """Собирает и сохраняет релевантные статьи в БД."""
        pass
