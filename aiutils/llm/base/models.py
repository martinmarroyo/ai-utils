from pydantic import BaseModel, Field
from typing import Optional
from typing import Dict
from typing import Any

class AIBaseModel(BaseModel):
  id: str = Field(..., description="The ID of the model")
  input_token_price: Optional[float] = Field(None, description="The price per input token")
  output_token_price: Optional[float] = Field(None, description="The price per output token")