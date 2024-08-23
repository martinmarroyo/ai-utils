"""A collection of classes and methods that are meant to manage other processes"""
from aiutils.llm.base.interfaces import BaseAIChat
from aiutils.llm.base.models import ChatMessage
from requests import Session
from aiohttp import ClientSession
from typing import List

class ChatManager:
  """A class for managing calls to an AI Chat model."""
  def __init__(self, llm: BaseAIChat):
    self.llm = llm
    self._session = None
    self._async_session = None
  
  def __enter__(self):
    self._session = Session()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self._session.close()

  def chat(self, messages: List[ChatMessage]):
      if not self._session:
        self._session = Session()
      response = self.llm.send_prompt(self._session, messages)
      return response

  async def __aenter__(self):
    self._async_session = ClientSession()
    return self

  async def __aexit__(self, exc_type, exc_value, traceback):
    await self._async_session.close()
  
  async def async_chat(self, messages: List[ChatMessage]):
      if not self._async_session:
        self._async_session = ClientSession()
      response = await self.llm.async_send_prompt(self._async_session, messages)
      return response

  @property
  def session(self):
    return self._session

  @property
  def async_session(self):
    return self._async_session