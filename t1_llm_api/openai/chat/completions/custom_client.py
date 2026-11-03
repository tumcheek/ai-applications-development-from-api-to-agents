import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Chat Completions API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, providing more control over the HTTP layer and demonstrating
    how to interact with the API directly.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no choices.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        #TODO:
        # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        # - Prepare headers with authorization and content type
        # - Prepare message history with System prompt
        # - Execute post request to AI API (use `requests`)
        # - Parse response
        # - Print response to console
        # - Return ASSISTANT message
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        messages = messages[::]
        system_promt = Message(role=Role.SYSTEM, content=self._system_prompt)
        messages.insert(0, system_promt)
        messages = list(map(lambda message: message.to_dict(), messages))
        request_body = json.dumps({
            "model": self._model_name,
            "messages": messages
        })
        response = requests.post(self._endpoint, headers=headers, data=request_body)
        response.raise_for_status()
        response_json = response.json()
        ai_answer = response_json.get("choices")[0].get("message").get("content")
        print(f"AI: {ai_answer}")
        ai_message = Message(role=Role.ASSISTANT, content=ai_answer)
        return ai_message

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed token-by-token using OpenAI's SSE format,
        with each chunk printed immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
        """
        #TODO:
        # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create (Streaming tab)
        # - Prepare headers with authorization and content type
        # - Prepare message history with System prompt
        # - Execute post request to AI API (use `aihttp`)
        # - Handle stream with chunks
        # - Parse response
        # - Print chunks to console
        # - Return ASSISTANT message
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        messages = messages[::]
        system_promt = Message(role=Role.SYSTEM, content=self._system_prompt)
        messages.insert(0, system_promt)
        messages = list(map(lambda message: message.to_dict(), messages))
        request_body = json.dumps({
            "model": self._model_name,
            "messages": messages,
            "stream": True
        })
        ai_answer = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=headers, data=request_body) as response:
                response.raise_for_status()
                print("AI:", end="")
                async for chunk in response.content:
                    text_line = chunk.decode("utf-8").strip()
                    if not text_line:
                        continue
                    if text_line.startswith("data: "):
                        content = text_line[6:]
                        if content == "[DONE]":
                            print()
                            break
                        try:
                            data_json = json.loads(content)
                            ai_content = data_json.get("choices")[0].get("delta").get("content") or ""
                            ai_answer += ai_content
                            print(ai_content, flush=True, end="")

                        except json.JSONDecodeError as e:
                            print("Some isssue occured")
                            raise e
        return Message(role=Role.ASSISTANT, content=ai_answer)

