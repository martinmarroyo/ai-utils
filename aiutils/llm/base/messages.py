from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator
from typing import Dict, Any, Optional, Annotated
import json

class ChatMessage(BaseModel):
  role: str = Field(..., description="The role of the message sender")
  content: Annotated[str, AfterValidator(json.dumps)] = Field(..., description="The content of the message")

  @classmethod
  def from_dict(cls, message_dict: Dict[str, Any]) -> "ChatMessage":
    return cls(**message_dict)

  def to_dict(self) -> Dict[str, Any]:
    return self.model_dump()
  
class ChatResponse(BaseModel):
  model: str = Field(..., description="The model that generated the response.")
  usage: Optional[Dict[str, float]] = Field(None, description="Token usage and cost incurred for the request.")
  response: str | BaseModel = Field(..., description="The response from the model.")

def format_for_chat(input_message: str, role: str = "user") -> ChatMessage:
  return ChatMessage(role=role, content=input_message)
