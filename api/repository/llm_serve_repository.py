from abc import ABC, abstractmethod


class LLMServeRepository(ABC):
    @abstractmethod
    def get_device(self):
        pass

    @abstractmethod
    def load_model(
        self, base_model_id: str, adapter_model_id, tokenizer_model_id, load_in_8bit
    ):
        pass

    @abstractmethod
    def predict(
        self,
        input,
        model,
        tokenizer,
        max_new_tokens,
        top_p,
        temperature,
        top_k,
        num_beams,
        repetition_penalty,
        do_sample,
        **kwargs
    ):
        pass

    @abstractmethod
    def get_embedding(self, input, model, tokenizer):
        pass
