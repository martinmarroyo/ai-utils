from pydantic import BaseModel, Field
from typing import Optional
from typing import Dict
from typing import Any

class ChatMessage(BaseModel):
  role: str = Field(..., description="The role of the message sender")
  content: str = Field(..., description="The content of the message")

  @classmethod
  def from_dict(cls, message_dict: Dict[str, Any]) -> "ChatMessage":
    return cls(**message_dict)

  def to_dict(self) -> Dict[str, Any]:
    return self.dict()
  
class ChatResponse(BaseModel):
  model: str = Field(..., description="The model that generated the response.")
  usage: Optional[Dict[str, float]] = Field(None, description="Token usage and cost incurred for the request.")
  response: str | BaseModel = Field(..., description="The response from the model.")

class AIBaseModel(BaseModel):
  id: str = Field(..., description="The ID of the model")
  input_token_price: Optional[float] = Field(None, description="The price per input token")
  output_token_price: Optional[float] = Field(None, description="The price per output token")