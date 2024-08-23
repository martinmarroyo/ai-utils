from aiutils.llm.base.messages import ChatMessage
from aiutils.llm.base.models import AIBaseModel
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class GPT4oMini(AIBaseModel):
  id: Literal["gpt-4o-mini"] = "gpt-4o-mini"
  input_token_price: float = 0.15 / (10 ** 6)
  output_token_price: float = 0.60 / (10 ** 6)

class GPT4o(AIBaseModel):
  id: Literal["gpt-4o"]
  input_token_price: float = 5.00 / (10 ** 6)
  output_token_price: float = 15.00 / (10 ** 6)

class OpenAIChatRequest(BaseModel):
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
