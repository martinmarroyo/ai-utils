from pydantic import BaseModel, Field
from typing import Optional
from typing import Dict
from typing import Any

class AIBaseModel(BaseModel):
  """Represents an AI model and metadata features about its usage."""
  id: str = Field(..., description="The ID of the model")
  input_token_price: Optional[float] = Field(0.0, description="The price per input token")
  output_token_price: Optional[float] = Field(0.0, description="The price per output token")