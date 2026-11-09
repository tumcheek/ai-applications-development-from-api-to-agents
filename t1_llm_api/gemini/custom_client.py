import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Google Gemini API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Gemini's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no candidates.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The URL is constructed by appending ':generateContent' to the model endpoint.
            Uses 'x-goog-api-key' header for authentication.
            Response candidates contain content parts that are concatenated.
        """
        #TODO:
        # https://ai.google.dev/gemini-api/docs/text-generation
        # - Prepare headers with api key and content type
        # - Add System prompt
        # - Execute post request to AI API (use `requests`)
        # - Parse response
        # - Print response to console
        # - Return ASSISTANT message
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
            "Api-Revision": "2026-05-20"
        }
        request_body = json.dumps({
            "model": self._model_name,
            "input": self._convert_msg(messages),
            "system_instruction": self._system_prompt
        })

        response = requests.post(self._endpoint, data=request_body, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        steps = response_json.get("steps")
        ai_answer = ""
        for step in steps:
            if step.get("type") == "model_output":
                for content in step.get("content"):
                    if content.get("type") == "text":
                        ai_answer = content.get("text")

        print("AI:", ai_answer)
        return Message(role=Role.ASSISTANT, content=ai_answer)


    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Gemini's SSE format, with text chunks
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The URL is constructed with ':streamGenerateContent?alt=sse' endpoint.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each SSE chunk contains candidates with content parts.
            Each text chunk is printed to stdout as it arrives.
        """
        #TODO:
        # https://ai.google.dev/gemini-api/docs/text-generation
        # - Prepare headers with api key and content type
        # - Add System prompt
        # - Execute post request to AI API (use `aiohttp`)
        # - Handle stream with chunks
        # - Parse response
        # - Print chunks to console
        # - Return ASSISTANT message
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
            "Api-Revision": "2026-05-20"
        }
        request_body = json.dumps({
            "model": self._model_name,
            "input": self._convert_msg(messages),
            "system_instruction": self._system_prompt,
            "stream":True
        })
        event = None
        ai_content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, data=request_body, headers=headers) as response:
                response.raise_for_status()
                print("AI:", end="")
                async for line in response.content:
                    text_line = line.decode("utf-8").strip()
                    if text_line.startswith("event: "):
                        event = text_line[7:]
                    elif text_line.startswith("data: "):
                        if event == "step.delta":
                            content_json = json.loads(text_line[6:])
                            content_delta = content_json.get("delta").get("text") or ""
                            ai_content += content_delta
                            print(content_delta, flush=True, end="")
                    elif not text_line:
                        event = None
        print()
                        
        
        return Message(role=Role.ASSISTANT, content=ai_content)

    def _convert_msg(self, messages: list[Message]) -> list[dict]:
        inputs = list(map(lambda ms: ms.to_dict(), messages))
        for message in inputs:
            role = message.get("role")
            text = message.get("content")
            message["type"] = "user_input" if role == Role.USER else "model_output"
            message["content"] = [
                {
                    "type": "text",
                    "text": text
                }
            ]
            del message["role"]
        return inputs