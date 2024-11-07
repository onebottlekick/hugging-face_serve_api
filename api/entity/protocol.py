import time
from typing import Any, Dict, List, Optional, Union

import shortuuid
from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    model: str = "huggingface_model"
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 40
    n: Optional[int] = 1
    max_tokens: Optional[int] = 128
    num_beams: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0
    user: Optional[str] = None
    do_sample: Optional[bool] = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    model: str = "huggingface_model"
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[Any]]
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    model: str = "huggingface_model"
    object: str = "list"
    data: List[Dict[str, Any]]


class CompletionRequest(BaseModel):
    prompt: Union[str, List[Any]]
    temperature: Optional[float] = 0.1
    n: Optional[int] = 1
    max_tokens: Optional[int] = 128
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 0.75
    top_k: Optional[int] = 40
    num_beams: Optional[int] = 1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0
    user: Optional[str] = None
    do_sample: Optional[bool] = True


class CompletionResponseChoice(BaseModel):
    index: int
    message: str


class CompletionResponse(BaseModel):
    model: Optional[str] = "huggingface_model"
    id: Optional[str] = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: Optional[str] = "text_completion"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    choices: List[CompletionResponseChoice]
