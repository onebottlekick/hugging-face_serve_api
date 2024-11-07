from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from peft import PeftModel

from api.repository.llm_serve_repository import LLMServeRepository


class LLMServeRepositoryImpl(LLMServeRepository):
    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(
        self,
        base_model_id: str,
        adapter_model_id: Optional[str] = None,
        tokenizer_model_id: Optional[str] = None,
        load_in_8bit: bool = True,
    ):
        device = self.get_device()

        if tokenizer_model_id is None:
            tokenizer_model_id = adapter_model_id
            if adapter_model_id is None:
                tokenizer_model_id = base_model_id

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
        if tokenizer.pad_token == None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device,
        )

        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(tokenizer)
        if model_vocab_size != tokenizer_vocab_size:
            assert tokenizer_vocab_size > model_vocab_size
            base_model.resize_token_embeddings(tokenizer_vocab_size)

        if adapter_model_id is not None:
            model = PeftModel.from_pretrained(
                base_model,
                adapter_model_id,
                torch_dtype=torch.float16,
                device_map=device,
            )
        else:
            model = base_model

        model.eval()

        return model, tokenizer

    def predict(
        self,
        input,
        model,
        tokenizer,
        max_new_tokens=8192,
        top_p=0.2,
        temperature=0.1,
        top_k=40,
        num_beams=4,
        repetition_penalty=1.0,
        do_sample=True,
        **kwargs
    ):
        device = self.get_device(device)
        prompt = " ".join([i.content for i in input])
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=float(repetition_penalty),
                attention_mask=inputs["attention_mask"],
            )
        sequences = generation_output.sequences[0]
        output = tokenizer.decode(sequences, skip_special_tokens=True)
        return output

    def get_embedding(self, input, model, tokenizer):
        device = self.get_device()
        encoding = tokenizer(input, padding=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        model_output = model(input_ids, attention_mask, output_hidden_state=True)
        data = model_output.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
        masked_embeddings = data * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        seq_length = torch.sum(mask, dim=1)
        embedding = sum_embeddings / seq_length
        normalized_embeddings = torch.nn.functional.normalize(embedding, p=2, dim=1)
        normalized_embeddings = normalized_embeddings.squeeze(0).tolist()
        return normalized_embeddings
