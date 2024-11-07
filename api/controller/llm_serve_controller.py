from fastapi import APIRouter, Depends

from api.entity.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    EmbeddingsRequest,
    EmbeddingsResponse,
)
from api.service.llm_serve_service_impl import LLMServeServiceImpl

llm_api_router = APIRouter()


async def inject_llm_api_service() -> LLMServeServiceImpl:
    return LLMServeServiceImpl()


@llm_api_router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    llm_api_service: LLMServeServiceImpl = Depends(inject_llm_api_service),
):
    messages = request.messages
    if isinstance(messages, str):
        messages = [ChatMessage(role="user", content=messages)]
    else:
        messages = [
            ChatMessage(role=message["role"], content=message["content"])
            for message in messages
        ]

    output = llm_api_service.completion(messages, request)
    choices = [
        ChatCompletionResponseChoice(
            index=0, message=ChatMessage(role="assistant", content=output)
        )
    ]
    return ChatCompletionResponse(choices=choices)


@llm_api_router.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingsRequest,
    llm_api_service: LLMServeServiceImpl = Depends(inject_llm_api_service),
):
    embedding = llm_api_service.embedding(request)
    data = [{"object": "embedding", "embedding": embedding, "index": 0}]
    return EmbeddingsResponse(data=data)
