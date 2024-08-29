from aiutils.llm.base.models import AIBaseModel
from aiutils.llm.base.interfaces import BaseAIChat
from aiutils.llm.base.messages import format_for_chat
from aiutils.llm.base.messages import ChatResponse
from aiutils.llm.base.messages import ChatMessage
from aiutils.llm.base.managers import ChatManager
from aiutils.llm.openai.models import OpenAIChatRequest
from aiutils.llm.openai.models import GPT4oMini
from aiutils.llm.openai.models import map_to_model
from aiutils.llm.openai.chat import OpenAIChat
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

OPENAI_API_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

class SimpleOpenAIChatbot(OpenAIChat):
  def __init__(self,
               chat_model: AIBaseModel | str = GPT4oMini,
               api_key: str = None,
               endpoint: str = OPENAI_API_CHAT_ENDPOINT,
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