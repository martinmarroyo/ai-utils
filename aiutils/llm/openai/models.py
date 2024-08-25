from aiutils.llm.base.messages import ChatMessage
from aiutils.llm.base.models import AIBaseModel
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class GPT4oMini(AIBaseModel):
  """Represents the gpt-4o-mini model from OpenAI."""
  id: Literal["gpt-4o-mini"] = "gpt-4o-mini"
  input_token_price: float = 0.15 / (10 ** 6)
  output_token_price: float = 0.60 / (10 ** 6)

class GPT4oLatest(AIBaseModel):
  """Represents the latest gpt-4o model from OpenAI."""
  id: Literal["gpt-4o-2024-08-06"] = "gpt-4o-2024-08-06"
  input_token_price: float = 2.50 / (10 ** 6)
  output_token_price: float = 10.00 / (10 ** 6)

class GPT4o(AIBaseModel):
  """Represents the current stable gpt-4o model from OpenAI."""
  id: Literal["gpt-4o", "gpt-4o-2024-05-13"] = "gpt-4o"
  input_token_price: float = 5.00 / (10 ** 6)
  output_token_price: float = 15.00 / (10 ** 6)

class ChatGPT4oLatest(AIBaseModel):
  """Represents the current stable gpt-4o model from OpenAI."""
  id: Literal["chatgpt-4o-latest"] = "chatgpt-4o-latest"
  input_token_price: float = 5.00 / (10 ** 6)
  output_token_price: float = 15.00 / (10 ** 6)

class GPT4(AIBaseModel):
  """Represents the current stable gpt-4 model from OpenAI."""
  id: Literal["gpt-4", "gpt-4-0613"] = "gpt-4"
  input_token_price: float = 30.00 / (10 ** 6)
  output_token_price: float = 60.00 / (10 ** 6)

class GPT4_32K(AIBaseModel):
  """Represents the current stable gpt-4-32k model from OpenAI."""
  id: Literal["gpt-4-32k"] = "gpt-4-32k"  
  input_token_price: float = 60.00 / (10 ** 6)
  output_token_price: float = 120.00 / (10 ** 6)

GPT4TurboModels = Literal[
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09"
]

class GPT4Turbo(AIBaseModel):
  """Represents the current stable gpt-4-turbo model from OpenAI."""
  id: GPT4TurboModels = "gpt-4-turbo"
  input_token_price: float = 10.00 / (10 ** 6)
  output_token_price: float = 30.00 / (10 ** 6)

class GPT35Turbo0125(AIBaseModel):
  """Represents the current stable gpt-3.5-turbo model from OpenAI."""
  id: Literal["gpt-3.5-turbo-0125"] = "gpt-3.5-turbo-0125"
  input_token_price: float = 0.50 / (10 ** 6)
  output_token_price: float = 1.50 / (10 ** 6)

class GPT35Turbo1106(AIBaseModel):
  """Represents the current stable gpt-3.5-turbo-1106 model from OpenAI."""
  id: Literal["gpt-3.5-turbo-1106"] = "gpt-3.5-turbo-1106"  
  input_token_price: float = 1.00 / (10 ** 6)
  output_token_price: float = 2.00 / (10 ** 6)

GPT35TurboInstructModels = Literal[
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301"
]

class GPT35TurboInstruct(AIBaseModel):
  """Represents the current stable gpt-3.5-turbo instruct models from OpenAI."""
  id: GPT35TurboInstructModels = "gpt-3.5-turbo-instruct"
  input_token_price: float = 1.50 / (10 ** 6)
  output_token_price: float = 2.00 / (10 ** 6)

class GPT35Turbo16k(AIBaseModel):
  """Represents the current stable gpt-3.5-turbo 16k model from OpenAI."""
  id: Literal["gpt-3.5-turbo-16k-0613"] = "gpt-3.5-turbo-16k-0613"
  input_token_price: float = 3.00 / (10 ** 6)
  output_token_price: float = 4.00 / (10 ** 6)

class OpenAIChatRequest(BaseModel):
  """The structure of a Chat Completions API request to OpenAI.

    See https://platform.openai.com/docs/api-reference/chat/create
    for details.
  """
  model: str = Field(..., description="The model to use for the chat")
  messages: List[ChatMessage] = Field(..., description="The messages to send to the model")
  logprobs: Optional[bool] = Field(False, description="Whether or not to include the logprobs in the response")
  max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate")
  temperature: Optional[float] = Field(0.0, description="The temperature to use for the model")
  response_format: Optional[Dict[str, Any]] = Field(None, description="The format to use for the response")
  seed: Optional[int] = Field(None, description="The seed to use to help ensure deterministic chat responses")
  top_p: Optional[float] = Field(None, description="The top_p to use for the model")
  stream: Optional[bool] = Field(False, description="Whether or not to stream the response")
  stream_options: Optional[Dict[str, Any]] = Field(None, description="The options to use for streaming")

OPENAI_MODEL_MAP = {
    "gpt-4-turbo": GPT4Turbo,
    "gpt-4-1106-preview": GPT4Turbo,
    "gpt-4-vision-preview": GPT4Turbo,
    "gpt-4-0125-preview": GPT4Turbo,
    "gpt-4-turbo-2024-04-09": GPT4Turbo,
    "gpt-4-32k": GPT4_32K,
    "gpt-4": GPT4, 
    "gpt-4-0613": GPT4,
    "chatgpt-4o-latest": ChatGPT4oLatest,
    "gpt-4o-mini": GPT4oMini,
    "gpt-4o-2024-08-06": GPT4oLatest,
    "gpt-4o": GPT4o,
    "gpt-3.5-turbo-0125": GPT35Turbo0125,
    "gpt-3.5-turbo-1106": GPT35Turbo1106,
    "gpt-3.5-turbo-instruct": GPT35TurboInstruct,
    "gpt-3.5-turbo-0613": GPT35TurboInstruct,
    "gpt-3.5-turbo-0301": GPT35TurboInstruct,
    "gpt-3.5-turbo-16k-0613": GPT35Turbo16k
}

def map_to_price_model(model: str|AIBaseModel) -> AIBaseModel:
  """Maps a model name to a model object to obtain pricing information"""
  if isinstance(model, str):
    if model not in OPENAI_MODEL_MAP:
      print(f"Cost tracking not available for {model}")
    return OPENAI_MODEL_MAP.get(model, model)
  return model