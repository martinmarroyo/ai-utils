"""A collection of interfaces for tools and connectors"""
from llm.base.models import ChatResponse
from llm.base.models import ChatMessage
from llm.base.models import AIBaseModel
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
    