"""A collection of interfaces for tools and connectors"""
from aiutils.llm.base.messages import ChatResponse
from aiutils.llm.base.messages import ChatMessage
from aiutils.llm.base.models import AIBaseModel
from abc import ABC, abstractmethod
from typing import List
from requests import Session
from aiohttp import ClientSession

class BaseAIChat(ABC):
  """A standardized interface for chat-related interactions with AI."""
  @abstractmethod
  def send_prompt(self, session: Session, messages: List[ChatMessage]) -> ChatResponse:
    pass
  @abstractmethod
  async def async_send_prompt(self, session: ClientSession, messages: List[ChatMessage]) -> ChatResponse:
    pass

  @abstractmethod
  def __init__(self, 
               model: str | AIBaseModel, 
               endpoint: str):
    self.model = model
    self.endpoint = endpoint
    