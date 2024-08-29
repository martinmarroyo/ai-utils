from requests.exceptions import HTTPError, Timeout, RequestException
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
import os
import requests
import aiohttp
import asyncio

OPENAI_FILES_ENDPOINT = "https://api.openai.com/v1/files"

class OpenAIFile(BaseModel):
  id: str
  object: str
  bytes: int
  created_at: int
  filename: str
  purpose: str
  status: Optional[str]
  status_details: Optional[str]

class OpenAIFileList(BaseModel):
  object: Literal["list"] = "list"
  data: List[OpenAIFile] = None

class OpenAIDeletedFile(BaseModel):
  id: str
  object: str
  deleted: bool

class OpenAIFileManager:
  def __init__(self,
               session: requests.Session = None,
               async_session: aiohttp.ClientSession = None,
               api_key: str = None,
               endpoint: str = OPENAI_FILES_ENDPOINT,
               additional_headers: Dict[str, str] = {}):
    self._session: requests.Session = session
    self._async_session: aiohttp.ClientSession = async_session
    self.api_key: str = api_key or os.environ.get("OPENAI_API_KEY")
    self._endpoint: str = endpoint
    self._headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}
    self._additional_headers: Dict[str, str] = additional_headers
    self._headers.update(self._additional_headers)
    self._file_list: OpenAIFileList = OpenAIFileList()

  def __enter__(self):
    self._init_session()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self._close_session()

  async def __aenter__(self):
    self._init_async_session()
    return self

  async def __aexit__(self, exc_type, exc_value, traceback):
    await self._close_async_session()
  
  # Initializers
  def _init_session(self):
    if not self._session:
      self._session = requests.Session()
    return self._session

  def _close_session(self):
    if self._session:
      self._session.close()

  def _init_async_session(self):
    if not self._async_session:
      self._async_session = aiohttp.ClientSession()
    return self._async_session

  async def _close_async_session(self):
    if self._async_session:
      await self._async_session.close() 
  
  # Validations
  def _check_if_can_retrieve_file_content(self, file_id: str):
    """Checks the cache to see if we can retrieve data for the given file_id."""
    file_info = list(filter(lambda f: f.id == file_id, self._file_list.data))
    if file_info and file_info[0].purpose == "assistants":
      print(f"File with id {file_id} is an assistant file and cannot be retrieved.")
      return False
    return True if file_info else None

  # Helper Methods
  def _retrieve_file_info_from_cache(self, file_id: str):
    for file in self._file_list.data:
      if file.id == file_id:
        return file
    return None

  def _retrieve_file_info_from_api(self, file_id: str):
    self._init_session()
    try:
      retrieved_file = self._session.get(headers=self._headers,
                                        url=f"{self._endpoint}/{file_id}")
      retrieved_file.raise_for_status()
      return OpenAIFile(**retrieved_file.json())
    except (HTTPError, Timeout, RequestException) as ex:
      print(f"Request failed with error: {retrieved_file.text}")
      raise ex
    except Exception as ex:
      print(f"An unexpected error occurred: {retrieved_file.text}")
      raise ex

  async def _async_retrieve_file_info_from_api(self, file_id: str):
    self._init_async_session()
    try:
      retrieved_file = await self._async_session.get(headers=self._headers,
                                        url=f"{self._endpoint}/{file_id}")
      retrieved_file.raise_for_status()
      response = await retrieved_file.json()
      return OpenAIFile(**response)
    except (aiohttp.ClientResponseError, aiohttp.ClientTimeout, aiohttp.ClientError) as ex:
      error_msg = await response.text()
      print(f"Request failed with error: {error_msg}")
      raise ex
    except Exception as ex:
      print(f"An unexpected exception occurred: {str(ex)}")
      raise ex

  def _retrieve_file_content_from_api(self,
                                      file_id: str,
                                      additional_headers: Dict[str, str] = {}):
    self._init_session()
    headers = self._headers.copy()
    headers.update(additional_headers)
    try:
      retrieved_file = self._session.get(headers=headers,
                                        url=f"{self._endpoint}/{file_id}/content")
      retrieved_file.raise_for_status()
      return retrieved_file.content
    except (HTTPError, Timeout, RequestException) as ex:
      print(f"Request failed with error: {retrieved_file.text}")
      raise ex
    except Exception as ex:
      print(f"An unexpected error occurred: {retrieved_file.text}")
      raise ex

  async def _async_retrieve_file_content_from_api(self,
                                      file_id: str,
                                      additional_headers: Dict[str, str] = {}):
    self._init_async_session()
    headers = self._headers.copy()
    headers.update(additional_headers)
    try:
      retrieved_file = await self._async_session.get(headers=headers,
                                        url=f"{self._endpoint}/{file_id}/content")
      retrieved_file.raise_for_status()
      response = await retrieved_file.content
      return response
    except (aiohttp.ClientResponseError, aiohttp.ClientTimeout, aiohttp.ClientError) as ex:
      error_msg = await response.text()
      print(f"Request failed with error: {error_msg}")
      raise ex
    except Exception as ex:
      print(f"An unexpected exception occurred: {str(ex)}")
      raise ex

  # Main Methods
  def upload_file(self,
                  file_path: str,
                  purpose: Literal["assistants", "vision", "fine-tune", "batch"],
                  additional_headers: Dict[str, str] = {}) -> OpenAIFile:
    """Upload a file that can be used across various endpoints. Individual files
    can be up to 512 MB, and the size of all files uploaded by one organization
    can be up to 100 GB.

    The Assistants API supports files up to 2 million tokens and of specific
    file types. See the Assistants Tools guide for details.

    The Fine-tuning API only supports .jsonl files. The input also has certain
    required formats for fine-tuning chat or completions models.

    The Batch API only supports .jsonl files up to 100 MB in size. The input
    also has a specific required format.

    See https://platform.openai.com/docs/api-reference/batch/request-input for
    more information on the Batch API format
    """
    self._init_session()
    headers = self._headers.copy()
    headers.update(additional_headers)
    try:
      request = self._session.post(
          url = self._endpoint,
          headers = headers,
          data = {"purpose": purpose},
          files = {"file": open(file_path, "rb")}
      )
      request.raise_for_status()
      # Update the file list
      self.get_file_list(update=True)
      return OpenAIFile(**request.json())
    except (HTTPError, Timeout, RequestException) as ex:
      print(f"Request failed with error: {request.text}")
      raise ex
    except Exception as ex:
      print(f"An unexpected error occurred: {request.text}")
      raise ex

  async def async_upload_file(self,
                  file_path: str,
                  purpose: Literal["assistants", "vision", "fine-tune", "batch"],
                  additional_headers: Dict[str, str] = {}) -> OpenAIFile:
    """The asynchronous version of `upload_file`.

    See `upload_file` for more information.
    """
    self._init_async_session()
    headers = self._headers.copy()
    headers.update(additional_headers)
    try:
      request = await self._async_session.post(
          url = self._endpoint,
          headers = headers,
          data = {"purpose": purpose, "file": open(file_path, "rb")}
      )
      request.raise_for_status()
      response = await request.json()
      # Update the file list
      await self.async_get_file_list(update=True)
      return OpenAIFile(**response)
    except (aiohttp.ClientResponseError, aiohttp.ClientTimeout, aiohttp.ClientError) as ex:
      error_msg = await response.text()
      print(f"Request failed with error: {error_msg}")
      raise ex
    except Exception as ex:
      print(f"An unexpected exception occurred: {str(ex)}")
      raise ex

  def get_file_list(self, update: bool = False) -> OpenAIFileList:
    """Returns a list of files that belong to the user's organization.

    You must set `update=True` in order to update the file list from the API.
    """
    if update:
      # Update from the API only when requested. Otherwise, hit the cache
      self._init_session()
      try:
        request = self._session.get(
            url = self._endpoint,
            headers = self._headers
        )
        request.raise_for_status()
        self._file_list = OpenAIFileList(**request.json())
      except (HTTPError, Timeout, RequestException) as ex:
        print(f"Request failed with error: {request.text}")
        raise ex
      except Exception as ex:
        print(f"An unexpected error occurred: {request.text}")
        raise ex
    return self._file_list

  async def async_get_file_list(self, update: bool = False) -> OpenAIFileList:
    """The asynchronous version of `get_file_list`.

    Returns a list of files that belong to the user's organization.

    You must set `update=True` in order to update the file list from the API.
    """
    if update:
      # Update from the API only when requested. Otherwise, hit the cache
      self._init_async_session()
      try:
        request = await self._async_session.get(headers=self._headers, url=self._endpoint)
        request.raise_for_status()
        response = await request.json()
        self._file_list = OpenAIFileList(**response)
      except (aiohttp.ClientResponseError, aiohttp.ClientTimeout, aiohttp.ClientError) as ex:
        error_msg = await response.text()
        print(f"Request failed with error: {error_msg}")
        raise ex
      except Exception as ex:
        print(f"An unexpected exception occurred: {str(ex)}")
        raise ex
    return self._file_list

  def retrieve_file_info(self, file_id: str):
    """Returns information about a specific file."""
    cached_file = self._retrieve_file_info_from_cache(file_id)
    if cached_file:
      return cached_file
    return self._retrieve_file_info_from_api(file_id)

  async def async_retrieve_file_info(self, file_id: str):
    """The asynchronous version of `retrieve_file_info`.

    Returns information about a specific file.
    """
    cached_file = self._retrieve_file_info_from_cache(file_id)
    if cached_file:
      return cached_file
    file_info = await self._async_retrieve_file_info_from_api(file_id)
    return file_info

  def retrieve_file_content(self, file_id: str, additional_headers: Dict[str, str] = {}):
    """Returns the contents of the specified file that was generated by an AI assistant."""
    # Check if file_id is in file list and that it does not have an "assistant" purpose
    can_retrieve = self._check_if_can_retrieve_file_content(file_id)
    if can_retrieve or can_retrieve is None:
      return self._retrieve_file_content_from_api(file_id, additional_headers)

  async def async_retrieve_file_content(self, file_id: str, additional_headers: Dict[str, str] = {}):
    """Returns the contents of the specified file that was generated by an AI assistant."""
    # Check if file_id is in file list and that it does not have an "assistant" purpose
    can_retrieve = self._check_if_can_retrieve_file_content(file_id)
    if can_retrieve or can_retrieve is None:
      file_content = await self._async_retrieve_file_content_from_api(file_id,
                                                                      additional_headers)
      return file_content

  def delete_file(self, 
                  file_id: str, 
                  additional_headers: Dict[str, str] = {}
                  ) -> OpenAIDeletedFile:
    """Deletes the file with the given file_id."""
    self._init_session()
    headers = self._headers.copy()
    headers.update(additional_headers)
    try:
      deleted_file = self._session.delete(headers=headers,
                                          url=f"{self._endpoint}/{file_id}")
      deleted_file.raise_for_status()
      self.get_file_list(update=True) # Update the file list
      return OpenAIDeletedFile(**deleted_file.json())
    except (HTTPError, Timeout, RequestException) as ex:
      print(f"Request failed with error: {deleted_file.text}")
      raise ex
    except Exception as ex:
      print(f"An unexpected error occurred: {deleted_file.text}")
      raise ex

  def delete_all_files(self) -> List[OpenAIDeletedFile]:
    """Deletes all files in the file list."""
    deletions = [self.delete_file(f.id) for f in self._file_list.data]
    return deletions

  async def async_delete_file(self, 
                              file_id: str, 
                              additional_headers: Dict[str, str] = {}
                              ) -> OpenAIDeletedFile:
    """Deletes the file with the given file_id."""
    self._init_async_session()
    headers = self._headers.copy()
    headers.update(additional_headers)
    try:
      deleted_file = await self._async_session.delete(headers=headers,
                                          url=f"{self._endpoint}/{file_id}")
      deleted_file.raise_for_status()
      response = await deleted_file.json()
      await self.async_get_file_list(update=True) # Update the file list
      return OpenAIDeletedFile(**response)
    except (aiohttp.ClientResponseError, aiohttp.ClientTimeout, aiohttp.ClientError) as ex:
      error_msg = await response.text()
      print(f"Request failed with error: {error_msg}")
      raise ex
    except Exception as ex:
      print(f"An unexpected exception occurred: {str(ex)}")
      raise ex

  async def async_delete_all_files(self) -> List[OpenAIDeletedFile]:
    """Deletes all files in the file list."""
    coros = [self.async_delete_file(f.id) for f in self._file_list.data]
    deletions = await asyncio.gather(*coros)
    return deletions

  @property
  def file_list(self):
    return self._file_list.data

  @property
  def file_count(self):
    return len(self._file_list.data)