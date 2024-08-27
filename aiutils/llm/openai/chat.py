from aiutils.llm.base.models import AIBaseModel
from aiutils.llm.base.interfaces import BaseAIChat
from aiutils.llm.base.messages import format_for_chat
from aiutils.llm.base.messages import ChatResponse
from aiutils.llm.base.messages import ChatMessage
from aiutils.llm.base.managers import ChatManager
from aiutils.llm.openai.models import OpenAIChatRequest
from aiutils.llm.openai.models import GPT4oMini
from aiutils.llm.openai.models import map_to_model
from requests.exceptions import HTTPError, Timeout, RequestException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from aiohttp import ClientSession
from io import StringIO
import os
import json
import time
import requests
import aiohttp
import base64

class BaseOpenAIChat(BaseAIChat):
  """A class for interacting with the OpenAI Chat API."""
  def __init__(self,
               chat_model: AIBaseModel | str = GPT4oMini,
               api_key: str = None,
               endpoint: str = "https://api.openai.com/v1/chat/completions",
               stream: bool = False,
               system_prompt: str = "You are a helpful assistant",
               prompt_template: str = None,
               response_format: Optional[Dict[str, Any] | BaseModel] = None,
               temperature: float = 0.0,
               logprobs: bool = False,
               top_p: float = None,
               seed: int = None,
               max_tokens: int = None,
               headers: Dict[str, str] = None):
    self.api_key: str = os.environ.get("OPENAI_API_KEY", api_key)
    self.chat_model: str | AIBaseModel = map_to_model(chat_model)
    self.model_id: str = self.chat_model.id
    self.system_prompt: str = system_prompt
    self.prompt_template: str = prompt_template
    self.temperature: float = temperature
    self.top_p: float = top_p
    self.max_tokens: int = max_tokens
    self.headers: Dict[str, str] = headers or self._generate_default_headers()
    self.stream: bool = stream
    self.stream_options: Optional[Dict[str, Any]] = ({"include_usage": True}
                                                     if stream else None)
    self.response_format: Optional[Dict[str, Any]] = response_format
    self.seed: Optional[int] = seed
    self.endpoint: str = endpoint
    self.logprobs: bool = logprobs

  def send_prompt(self,
                  session: requests.Session,
                  messages: List[ChatMessage],
                  sys_prompt: str = None) -> ChatResponse:
    """Sends a synchronous prompt to the OpenAI API Chat endpoint."""
    if self.stream:
       raise Exception("Streaming not supported for synchronous requests. "
                       "Set `stream=False` to use this method.")

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
      raise ex
    except Exception as ex:
      print(f"An unexpected error occurred: {response.text}")
      raise ex

  async def async_send_prompt(self,
                        session: aiohttp.ClientSession,
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

    except (aiohttp.ClientResponseError, aiohttp.ClientTimeout, aiohttp.ClientError) as ex:
      error_msg = await response.text()
      print(f"Request failed with error: {error_msg}")
      raise ex
    except Exception as ex:
      print(f"An unexpected exception occurred: {str(ex)}")
      raise ex
  
  def stream_prompt(self,
                    session: requests.Session,
                    messages: List[ChatMessage],
                    sys_prompt: str = None):
    """Streams the response from the OpenAI API Chat endpoint."""
    request = self._prepare_request(messages=messages, sys_prompt=sys_prompt)
    try:
      response = session.post(
        self.endpoint,
        headers=self.headers,
        data=request
      )
      # Check response status and send results back to user
      response.raise_for_status()
      for line in response.iter_lines():
        if line:
          try:
            result = json.loads(line.decode("utf-8").replace("data: ", ""))
            yield result
          except TypeError as type_err:
            continue
    except (HTTPError, Timeout, RequestException) as ex:
      pass
      # print(f"Request failed with error: {response.text}")
    except Exception as ex:
      pass
      # print(f"An unexpected error occurred: {response.text}")

  def _apply_prompt_template(self, messages: List[ChatMessage]):
    """Apply the prompt template to the messages."""
    for message in messages:
      if message.role in ("user", "human"):
        try:
          message.content = self.prompt_template.format(
                            **json.loads(message.content))
        except Exception:
          raise ValueError(
              f"Input: {message.content} does not match provided "
              "prompt template.")
        
  def _prepare_request(self, messages: List[ChatMessage], sys_prompt: str = None):
    """Prepare the request to the OpenAI API."""
    # Apply prompt template to messages if provided. Raise error if input format
    # does not match the supplied template
    if self.prompt_template:
      self._apply_prompt_template(messages)

    # Configure output format for the LLM response
    output_format = self.response_format
    if issubclass(self._check_output_format(), BaseModel):
        output_format = self._structure_model_schema()
    # Set up a request for the chat completions API
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
      logprobs=self.logprobs,
      stream=self.stream,
      stream_options=self.stream_options
    ).model_dump_json()

    return request

  def _prepare_response(self, response: Dict[str, Any]) -> ChatResponse:
    """Prepare the response from the OpenAI API."""
    chat_response = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
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
    inner_models = schema.get("$defs", {})
    for model in inner_models:
      inner_models[model]["additionalProperties"] = False
      inner_models[model]["required"] = list(
          inner_models[model]["properties"].keys())
    if inner_models:
      del schema["$defs"]
      schema["schema"]["$defs"] = inner_models
    schema["schema"].update({"additionalProperties": False,})
    schema["strict"] = True
    if schema.get("required"):
      del schema["required"]
    return {
          "type": "json_schema",
          "json_schema": schema
    }

  def _calculate_cost(self, usage: Dict[str, int]) -> Dict[str, float]:
    """Calculate the cost of the request."""
    input_cost = usage.get("prompt_tokens", 0) * self.chat_model.input_token_price
    output_cost = usage.get("completion_tokens", 0) * self.chat_model.output_token_price
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
               system_prompt: str = "You are a helpful assistant",
               prompt_template: str = None,
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
                              system_prompt=system_prompt,
                              prompt_template=prompt_template,
                              response_format=response_format,
                              temperature=temperature,
                              logprobs=logprobs,
                              top_p=top_p,
                              seed=seed,
                              max_tokens=max_tokens,
                              headers=headers)
    ChatManager.__init__(self, llm=self)

  def stream_chat(self, messages: List[ChatMessage] | Dict[str, Any] | str):
    if not isinstance(messages, list):
      messages = [ChatMessage(role="user", content=messages)]

    if self.session is None:
      self.session = requests.Session()

    return self.stream_prompt(self.session, messages, self.system_prompt)


class SimpleOpenAIChatbot(OpenAIChat):
  def __init__(self,
               chat_model: AIBaseModel | str = GPT4oMini,
               api_key: str = None,
               endpoint: str = "https://api.openai.com/v1/chat/completions",
               stream: bool = False,
               system_prompt: str = "You are a helpful assistant",
               prompt_template: str = None,
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
                    system_prompt=system_prompt,
                    response_format=response_format,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_p=top_p,
                    seed=seed,
                    max_tokens=max_tokens,
                    headers=headers)

    self.message_history = [
        ChatMessage(role="system", content=system_prompt or self.system_prompt)
    ]
    self.MESSAGE_BUFFER_SIZE = message_buffer_size
    self.chat_prompt_template = prompt_template
    summary_prompt = ("Create a one or two paragraph summary of the given "
                      "chat history so that we can understand the conversation")
    self.summarizer = OpenAIChat(chat_model=chat_model,
                                 system_prompt=summary_prompt)
    self.usage = {}
  
  def _tally_total_cost(self, current_usage: Dict[str, float]):
    """Tally the total cost the current conversation."""
    current_cost = self.usage
    for item in current_usage:
      if item in current_cost:
        current_cost[item] += current_usage[item]
        continue
      current_cost[item] = current_usage[item]
    self.usage = current_cost 

  def _simple_message_compaction(self):
      """Compacts message history once the MESSAGE_BUFFER_SIZE is reached."""
      try:
        summary = self.summarizer.chat(self.message_history)
        compacted_history = [
            self.message_history[0],
            ChatMessage(role = "assistant", content=summary.response)]
        self.message_history = compacted_history
      except Exception as ex:
        print(f"An unexpected exception occurred during compaction: {str(ex)}")

  def conversation(self):
      """Starts a conversation loop with the user."""
      while True:
        with self as llm:
          question = input("Ask me anything!\t")
          if question.lower() in ("quit", "exit", "bye", "/q"):
            print("Goodbye!")
            break

          if len(self.message_history) >= self.MESSAGE_BUFFER_SIZE:
            self._simple_message_compaction()

          if self.chat_prompt_template:
            try:
              question = self.chat_prompt_template.format(content=question)
            except KeyError:
              question = f"{self.chat_prompt_template}\n{question}"
              
          self.message_history.append(format_for_chat(question))
          # Handle streaming messages
          if self.stream:
            token_buffer = StringIO()
            usage = {}
            for elem in llm.stream_chat(self.message_history):
              if elem:
                if elem.get("usage"):
                  usage = elem["usage"]
                  continue
                token = elem["choices"][0]["delta"].get("content", "")
                token_buffer.write(token)
                print(token, end="")
                time.sleep(0.025)
            # Add cost to usage if cost model is available
            if isinstance(self.chat_model, AIBaseModel):
              cost =  self._calculate_cost(usage)
              usage = {**usage, **cost}
            self._tally_total_cost(usage)
            response = ChatResponse(model=self.model_id, 
                                  usage=usage, 
                                  response=token_buffer.getvalue())
          else:
            # Handle non-streaming messages
            response = llm.chat(self.message_history)
            print(response.response)
            self._tally_total_cost(response.usage)
          self.message_history.append(ChatMessage(role="assistant",
                                                  content=response.response))
          print("\n\n--------------------------------------------")


def create_image_url_from_file(filepath: str):
  # Check that filepath exists
  assert os.path.exists(filepath), f"{filepath} does not exist"
  # Open file from path
  with open(filepath, "rb") as f:
    # Read and b64 encode file
    image_file = base64.b64encode(f.read()).decode('utf-8')
  # Create the image_url
  image_url = f"data:image/jpeg;base64,{image_file}"
  return image_url

def create_chat_message_with_images(text: str, image_url: str | List[str]):
  """Creates a chat message that includes both text and one or more images."""
  text_format = {"type": "text", "text": text}
  if isinstance(image_url, str):
    image_url = [image_url]
  image_format = [{"type": "image_url", "image_url": {"url": img}} for img in image_url]
  content = [text_format] + image_format
  return ChatMessage(role="user", content=json.dumps(content))