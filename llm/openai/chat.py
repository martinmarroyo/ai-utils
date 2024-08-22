from llm.openai.models import OpenAIModel
from llm.openai.models import OpenAIChatRequest
from llm.base.base_models import ChatMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import os
import requests
import json


class OpenAIChat:

  def __init__(self,
               api_key: str = None,
               endpoint: str = "https://api.openai.com/v1/chat/completions", 
               stream: bool = False,
               stream_options: Dict[str, Any] = None,
               chat_model: str | Any = "gpt-4o-mini",
               system_prompt: str = "You are a helpful assistant",
               response_format: Dict[str, Any] | BaseModel = None,
               temperature: float = 0.0,
               logprobs: bool = False,
               top_p: float = None,
               seed: int = None, 
               max_tokens: int = None,  
               headers: Dict[str, str] = None):
    self.api_key: str = os.environ.get("OPENAI_API_KEY", api_key)
    self.chat_model: str | OpenAIModel = chat_model
    self.model_id: str = chat_model.id if isinstance(chat_model, OpenAIModel) else chat_model
    self.system_prompt: str = system_prompt
    self.temperature: float = temperature
    self.top_p: float = top_p
    self.max_tokens: int = max_tokens
    self.headers: Dict[str, str] = headers or self._generate_default_headers()
    self.stream: bool = stream
    self.stream_options: Optional[Dict[str, Any]] = stream_options
    self.response_format: Optional[Dict[str, Any]] = response_format
    self.seed: Optional[int] = seed
    self.endpoint: str = endpoint
    self.logprobs: bool = logprobs


  def send_prompt(self, session: requests.Session, messages: List[ChatMessage]) -> str:
    request = OpenAIChatRequest(
      model=self.model_id,
      messages=[
        ChatMessage(role="system", content=self.system_prompt),
        *messages
      ],
      temperature=self.temperature,
      max_tokens=self.max_tokens,
      top_p=self.top_p,
      stream=self.stream,
      stream_options=self.stream_options,
      response_format=self.response_format,
      seed=self.seed,
      logprobs=self.logprobs
    ).model_dump_json()
    response = requests.post(
      self.endpoint,
      headers=self.headers,
      data=request
    ).json()


    if isinstance(self.chat_model, OpenAIModel):
      cost =  self._calculate_usage(response["usage"])
    
    return cost, response

   

  def _calculate_usage(self, usage: Dict[str, int]) -> Dict[str, float]:
    
    input_cost = usage["prompt_tokens"] * self.chat_model.input_token_price
    output_cost = usage["completion_tokens"] * self.chat_model.output_token_price
    total_cost = input_cost + output_cost
    return dict(**usage,
                input_cost=input_cost, 
                output_cost=output_cost, 
                total_cost=total_cost)

  def _generate_default_headers(self) -> Dict[str, str]:
    return {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {self.api_key}"
    }
