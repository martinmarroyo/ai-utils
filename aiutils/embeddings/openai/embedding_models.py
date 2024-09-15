from typing import Dict, List, Union, Literal, Optional
from ...llm.base.models import AIBaseModel
from pydantic import BaseModel
import requests
import aiohttp
import os

OPENAI_EMBEDDINGS_ENDPOINT = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

class TextEmbedding3Small(AIBaseModel):
  id: Literal["text-embedding-3-small"] = "text-embedding-3-small"
  input_token_price: float = 0.02 / (10**6)

class TextEmbedding3Large(AIBaseModel):
  id: Literal["text-embedding-3-large"] = "text-embedding-3-large"
  input_token_price: float = 0.13 / (10**6)

class AdaV2(AIBaseModel):
  id: Literal["text-embedding-ada-002"] = "text-embedding-ada-002"
  input_token_price: float = 0.10 / (10**6)

class OpenAIEmbeddingRequest(BaseModel):
  input: Union[str, List[str]]
  model: str
  encoding_format: Literal["float", "base64"] = "float"
  dimensions: Optional[int] = None
  user: Optional[str] = None

class OpenAIEmbeddingResponse(BaseModel):
  object: Literal["embedding"]
  embedding: List[float]
  index: int
  usage: Optional[Dict[str, float|int]] = None

def text_to_model(text: str) -> AIBaseModel:
  if not isinstance(text, str):
    return text
  if text == "text-embedding-3-small":
    return TextEmbedding3Small()
  elif text == "text-embedding-3-large":
    return TextEmbedding3Large()
  elif text == "text-embedding-ada-002":
    return AdaV2()
  else:
    print(f"Cost tracking not available for {text}")
    return AIBaseModel(id=text)

class OpenAIEmbedding:
  def __init__(self, 
               api_key:str=None,
               endpoint:str=OPENAI_EMBEDDINGS_ENDPOINT, 
               model_name:AIBaseModel=TextEmbedding3Small,
               sync_session:requests.Session=None,
               async_session:aiohttp.ClientSession=None,
               headers:Dict[str,str]=None,
               dimensions:int=None,
               encoding_format:str="float",
               cost_tracking:bool=True):
    try:
      self.api_key = api_key or os.environ["OPENAI_API_KEY"]
    except KeyError:
      raise ValueError("API key not found. Please set it "
                       "as an environment variable or pass "
                       "it as an argument.")

    self.model = text_to_model(model_name)()
    self._endpoint = endpoint or OPENAI_EMBEDDINGS_ENDPOINT
    self._sync_session = sync_session 
    self._init_session()
    self._async_session = async_session 
    self._async_init_session()
    self._headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
      }
    if headers:
      self._headers.update(headers)
    if dimensions:
      if self.model.id == "text-embedding-ada-002":
        raise ValueError("Dimensions are not supported for Ada v2. "
                         "Please choose a different model or set dimensions=None.")
    self.dimensions = dimensions
    self.encoding_format = encoding_format
    self.cost_tracking_enabled = cost_tracking
    self.total_spend_usd = None

  def _init_session(self):
    if not self._sync_session:
      self._sync_session = requests.Session()
    return self._sync_session

  def _close_session(self):
    if self._sync_session:
      self._sync_session.close()

  def __enter__(self):
    if not self._sync_session:
      self._init_session()
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    self._close_session()

  def _async_init_session(self):
    if not self._async_session:
      self._async_session = aiohttp.ClientSession()
    return self._async_session

  async def async_close_session(self):
    if self._async_session:
      await self._async_session.close()

  async def __aenter__(self):
    if not self._async_session:
      await self._async_init_session()
    return self

  async def __aexit__(self, exc_type, exc_value, traceback):
    await self.async_close_session()
    
  def embed(self, content: str):
    request = self._create_request(content)
    response = self._sync_session.post(url=self._endpoint,**request)
    response.raise_for_status()
    payload = response.json()
    usage = self._calculate_cost(payload["usage"])
    embedding_response = payload["data"][0]
    embedding_response["usage"] = usage
    return OpenAIEmbeddingResponse(**embedding_response)

  async def async_embed(self, content:str):
    request = self._create_request(content)
    response = await self._async_session.post(url=self._endpoint,**request)
    response.raise_for_status()
    payload = await response.json()
    usage = self._calculate_cost(payload["usage"])
    embedding_response = payload["data"][0]
    embedding_response["usage"] = usage
    return OpenAIEmbeddingResponse(**embedding_response)
  
  def _calculate_cost(self, usage: Dict[str, int]):
    cost = usage["total_tokens"] * self.model.input_token_price
    if self.cost_tracking_enabled:
      if not self.total_spend_usd:
        self.total_spend_usd = 0
      self.total_spend_usd += cost

    usage["cost"] = cost
    return usage

  def _create_request(self, content: str):
    data_request = OpenAIEmbeddingRequest(
      input=content,
      model=self.model.id,
      encoding_format=self.encoding_format
    )
    if self.dimensions:
      data_request.dimensions = self.dimensions
    
    request = dict(headers=self._headers, 
                   data=data_request.model_dump_json(exclude_none=True))
    return request