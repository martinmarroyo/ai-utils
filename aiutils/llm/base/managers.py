"""A collection of classes and methods that are meant to manage other processes"""
from aiutils.llm.base.interfaces import BaseAIChat
from aiutils.llm.base.messages import ChatMessage
from requests.exceptions import HTTPError, Timeout, RequestException
from aiohttp.client_exceptions import ClientResponseError, ClientError
from aiohttp.client import ClientTimeout
from requests import Session
from aiohttp import ClientSession
from typing import List, Dict, Any
import asyncio
import time
import json

class ChatManager:
  """A class for managing calls to an AI Chat model."""
  def __init__(self, 
               llm: BaseAIChat, 
               max_retries: int = 10, 
               delay: int = 1, 
               penalty: int = 2):
    self.llm = llm
    self._session = None
    self._async_session = None
    self._max_retries = max_retries
    self._delay = delay
    self._penalty = penalty
  
  def __enter__(self):
    self._session = Session()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self._session.close()

  def chat(self, messages: List[ChatMessage] | Dict[str, Any] | str):
      """Synchronous chat with the given LLM"""
      if not self._session:
        self._session = Session()

      if not isinstance(messages, list):
        messages = [ChatMessage(role="user", content=messages)]

      retries = 0
      penalty_wait = 0
      while retries < self._max_retries:
        try:
          response = self.llm.send_prompt(self._session, messages)
          return response
        except (HTTPError, Timeout, RequestException) as req_err:
          retries += 1
          if retries == self._max_retries:
            print("Retries exhausted. Please check your rate limits "
                  "and try again later.")
            raise req_err
          time.sleep(self._delay + penalty_wait)
          penalty_wait += self._penalty
        except Exception as ex:
          print(f"An unexpected error occurred: {str(ex)}")
          raise ex

  async def __aenter__(self):
    self._async_session = ClientSession()
    return self

  async def __aexit__(self, exc_type, exc_value, traceback):
    await self._async_session.close()
  
  async def async_chat(self, messages: List[ChatMessage] | Dict[str, Any] | str):
      """Asynchronous chat with the given LLM"""
      if not self._async_session:
        self._async_session = ClientSession()

      if not isinstance(messages, list):
        messages = [ChatMessage(role="user", content=messages)]

      retries = 0
      penalty_wait = 0
      while retries < self._max_retries:
        try:
          response = await self.llm.async_send_prompt(self._async_session, messages)
          return response
        except (ClientResponseError, ClientTimeout, ClientError) as ex:
          retries += 1
          if retries == self._max_retries:
            print("Retries exhausted. Please check your rate limits "
                  "and try again later.")
            raise ex
          await asyncio.sleep(self._delay + penalty_wait)
          penalty_wait += self._penalty

  @property
  def session(self):
    return self._session

  @property
  def async_session(self):
    return self._async_session

  @property
  def max_retries(self):
    return self._max_retries

  @max_retries.setter
  def max_retries(self, value):
    self._max_retries = value
    
  @property
  def delay(self):
    return self._delay

  @delay.setter
  def delay(self, value):
    self._delay = value

  @property
  def penalty(self):
    return self._penalty

  @penalty.setter
  def penalty(self, value):
    self._penalty = value 