from api.service.llm_serve_service import LLMServeService
from api.repository.llm_serve_repository_impl import LLMServeRepositoryImpl


class LLMServeServiceImpl(LLMServeService):
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    def __init__(self):
        self.llm_repository = LLMServeRepositoryImpl()

    def completion(self, messages, request):
        model, tokenizer = self.llm_repository.load_model(self.MODEL_ID)
        output = self.llm_repository.predict(
            messages,
            model,
            tokenizer,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            temperature=request.temperature,
            top_k=request.top_k,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        return output

    def embedding(self, request):
        model, tokenizer = self.llm_repository.load_model(self.MODEL_ID)
        embedding = self.llm_repository.get_embedding(request.input, model, tokenizer)
        return embedding
