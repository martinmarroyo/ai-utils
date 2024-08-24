from aiutils.llm.base.models import AIBaseModel
from aiutils.llm.base.interfaces import BaseAIChat
from aiutils.llm.base.messages import format_for_chat
from aiutils.llm.base.messages import ChatResponse
from aiutils.llm.base.messages import ChatMessage
from aiutils.llm.base.managers import ChatManager
from aiutils.llm.openai.models import OpenAIChatRequest
from aiutils.llm.openai.models import GPT4oMini
from requests.exceptions import HTTPError, Timeout, RequestException
from aiohttp.client_exceptions import ClientResponseError, ClientError
from aiohttp.client import ClientTimeout
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from aiohttp import ClientSession
import os
import requests

class BaseOpenAIChat(BaseAIChat):
  """A class for interacting with the OpenAI Chat API."""
  def __init__(self,
               chat_model: AIBaseModel | str = GPT4oMini,
               api_key: str = None,
               endpoint: str = "https://api.openai.com/v1/chat/completions",
               stream: bool = False,
               stream_options: Dict[str, Any] = None,
               system_prompt: str = "You are a helpful assistant",
               response_format: Optional[Dict[str, Any] | BaseModel] = None,
               temperature: float = 0.0,
               logprobs: bool = False,
               top_p: float = None,
               seed: int = None,
               max_tokens: int = None,
               headers: Dict[str, str] = None):
    self.api_key: str = os.environ.get("OPENAI_API_KEY", api_key)
    self.chat_model: str | AIBaseModel = chat_model()
    self.model_id: str = (self.chat_model.id
                          if isinstance(self.chat_model, AIBaseModel)
                          else chat_model)
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

  def send_prompt(self,
                  session: requests.Session,
                  messages: List[ChatMessage],
                  sys_prompt: str = None) -> ChatResponse:
    """Sends a synchronous prompt to the OpenAI API Chat endpoint."""
    request = self._prepare_request(messages=messages, sys_prompt=sys_prompt)
    try:
      response = session.post(
        self.endpoint,
        headers=self.headers,
        data=request
      )
      # Check response status and send results back to user
      response.raise_for_status()
      result = self._prepare_response(response.json())

      return result
    except (HTTPError, Timeout, RequestException) as ex:
      print(f"Request failed with error: {response.text}")
      print(f"Request type: {type(request)}\nRequest: {request}")
      raise ex
    except Exception as ex:
      print(f"An unexpected error occurred: {str(ex)}")
      raise ex

  async def async_send_prompt(self,
                        session: ClientSession,
                        messages: List[ChatMessage],
                        sys_prompt: str = None) -> ChatResponse:
    """Sends a prompt to the OpenAI API asynchronously and returns the response."""
    try:
      request = self._prepare_request(messages=messages, sys_prompt=sys_prompt)
      response = await session.post(
        self.endpoint,
        headers=self.headers,
        data=request
      )
      response.raise_for_status()
      result = self._prepare_response(await response.json())
      return result

    except (ClientResponseError, ClientTimeout, ClientError) as ex:
      error_msg = await response.text()
      print(f"Request failed with error: {error_msg}")
      print(f"Request type: {type(request)}\nRequest: {request}")
      raise ex
    
    except Exception as ex:
      print(f"An unexpected exception occurred: {str(ex)}")
      raise ex
    
  def _prepare_request(self, messages: List[ChatMessage], sys_prompt: str = None):
    """Prepare the request to the OpenAI API."""
    # Configure output format
    output_format = self.response_format
    if issubclass(self._check_output_format(), BaseModel):
        output_format = self._structure_model_schema()
    # Set up and send a synchronous request
    request = OpenAIChatRequest(
      model=self.model_id,
      messages=[
        ChatMessage(role="system", content=sys_prompt or self.system_prompt),
        *messages
      ],
      temperature=self.temperature,
      max_tokens=self.max_tokens,
      top_p=self.top_p,
      response_format=output_format,
      seed=self.seed,
      logprobs=self.logprobs
    ).model_dump_json()

    return request

  def _prepare_response(self, response: Dict[str, Any]) -> ChatResponse:
    """Prepare the response from the OpenAI API."""
    chat_response = response["choices"][0]["message"]["content"]
    usage = response["usage"]
    if issubclass(self._check_output_format(), BaseModel):
      chat_response = self.response_format.parse_raw(chat_response)

    if isinstance(self.chat_model, AIBaseModel):
      cost =  self._calculate_cost(usage)
      usage = {**usage, **cost}

    return ChatResponse(
      model=self.model_id,
      usage=usage,
      response=chat_response
    )

  def _check_output_format(self) -> BaseModel | Dict[str, Any]:
    if self.response_format and not isinstance(self.response_format, dict):
      assert issubclass(self.response_format, BaseModel), \
      "Response format must be a subclass of `BaseModel` or a dictionary."
      return BaseModel
    return dict

  def _structure_model_schema(self) -> Dict[str, Any]:
    """Structure the model schema for the response."""
    schema = self.response_format.model_json_schema()
    schema["name"] = schema.pop("title")
    schema["schema"] = {
        "type": schema.pop("type"),
        "properties": schema.pop("properties"),
        "additionalProperties": False,
        "required": list(self.response_format.model_fields.keys())
    }
    if "$defs" in schema:
      schema["schema"].update({"$defs": schema.pop("$defs")})
    schema["strict"] = True
    if schema.get("required"):
      del schema["required"]
    return {
          "type": "json_schema",
          "json_schema": schema
    }

  def _calculate_cost(self, usage: Dict[str, int]) -> Dict[str, float]:
    """Calculate the cost of the request."""
    input_cost = usage["prompt_tokens"] * self.chat_model.input_token_price
    output_cost = usage["completion_tokens"] * self.chat_model.output_token_price
    total_cost = input_cost + output_cost
    return dict(input_cost_usd=input_cost,
                output_cost_usd=output_cost,
                total_cost_usd=total_cost)

  def _generate_default_headers(self) -> Dict[str, str]:
    return {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {self.api_key}"
    }

  def update_output_structure(self, model: BaseModel | Dict[str, Any]):
    """Update the desired output structure of the model."""
    if isinstance(model, dict):
      assert model["type"] == "json_schema", \
      "Response format type must be of `json_schema` for structured output."
    
    self.response_format = model


class OpenAIChat(ChatManager, BaseOpenAIChat):

  def __init__(self, 
               chat_model: AIBaseModel | str = GPT4oMini,
               api_key: str = None,
               endpoint: str = "https://api.openai.com/v1/chat/completions",
               stream: bool = False,
               stream_options: Dict[str, Any] = None,
               system_prompt: str = "You are a helpful assistant",
               response_format: Optional[Dict[str, Any] | BaseModel] = None,
               temperature: float = 0.0,
               logprobs: bool = False,
               top_p: float = None,
               seed: int = None,
               max_tokens: int = None,
               headers: Dict[str, str] = None):
    
    BaseOpenAIChat.__init__(self, chat_model=chat_model,
                              api_key=api_key,
                              endpoint=endpoint,
                              stream=stream,
                              stream_options=stream_options,
                              system_prompt=system_prompt,
                              response_format=response_format,
                              temperature=temperature,
                              logprobs=logprobs,
                              top_p=top_p,
                              seed=seed,
                              max_tokens=max_tokens,
                              headers=headers)
    ChatManager.__init__(self, llm=self)


class SimpleOpenAIChatbot(OpenAIChat):
  def __init__(self, 
               chat_model: AIBaseModel | str = GPT4oMini,
               api_key: str = None,
               endpoint: str = "https://api.openai.com/v1/chat/completions",
               stream: bool = False,
               stream_options: Dict[str, Any] = None,
               system_prompt: str = "You are a helpful assistant",
               response_format: Optional[Dict[str, Any] | BaseModel] = None,
               temperature: float = 0.0,
               logprobs: bool = False,
               top_p: float = None,
               seed: int = None,
               max_tokens: int = None,
               headers: Dict[str, str] = None,
               message_buffer_size: int = 5):
    super().__init__(chat_model=chat_model,
                    api_key=api_key,
                    endpoint=endpoint,
                    stream=stream,
                    stream_options=stream_options,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_p=top_p,
                    seed=seed,
                    max_tokens=max_tokens,
                    headers=headers)
    
    self.message_history = [
        ChatMessage(role="system", 
                    content=system_prompt or self.system_prompt)
    ]
    self.MESSAGE_BUFFER_SIZE = message_buffer_size

  def _simple_message_compaction(self):
      print("Compacting messages...")
      raw_messages = [m.content for m in self.message_history]
      summary = self.chat("Summarize the following conversation in one to "
                            f"two paragraphs:\n\n{raw_messages}")
      compacted_history = [
          self.message_history[0], 
          ChatMessage(role = "assistant", content=summary.response)]
      print("Compaction completed!")
      self.message_history = compacted_history

  def conversation(self):
      while True:
        with self as llm:
          question = input("Ask me anything!\t")
          if question.lower() in ("quit", "exit", "bye", "/q"):
            print("Goodbye!")
            break

          if len(self.message_history) >= self.MESSAGE_BUFFER_SIZE:
            self._simple_message_compaction()

          self.message_history.append(format_for_chat(question))
          result = llm.chat(self.message_history)
          print(result.response, end=f"\n\n{'-' * 40}\n\n")
          self.message_history.append(ChatMessage(role="assistant", 
                                          content=result.response))