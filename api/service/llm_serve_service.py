from abc import ABC, abstractmethod


class LLMServeService(ABC):
    @abstractmethod
    def completion(self, messages, request):
        pass

    @abstractmethod
    def embedding(self, request):
        pass
