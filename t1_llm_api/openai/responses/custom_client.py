import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient

from pprint import pprint


class CustomOpenAIResponsesClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Responses API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with the Responses API directly
    and handle its unique event-based streaming format.
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
            ValueError: If the API response contains no output text.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Uses the Responses API format with 'instructions' and 'input' parameters.
            The response is printed to stdout before being returned.
        """
        #TODO:
        # https://developers.openai.com/api/docs/guides/text?lang=curl
        # - Prepare headers with authorization and content type
        # - Prepare input messages
        # - Execute post request to AI API (use `requests`)
        # - Parse response
        # - Print response to console
        # - Return ASSISTANT message
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        inputs = list(map(lambda ms: ms.to_dict(), messages))
        request_body = json.dumps({
            "model": self._model_name,
            "instructions": self._system_prompt,
            "input": inputs
        })
        response = requests.post(url=self._endpoint, data=request_body, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        ai_answer = "-"
        for content in response_json.get("output"):
            if content.get("type") == "message":
                ai_answer = content.get("content")[0].get("text")   
        print(f"AI: {ai_answer}")
        return Message(role=Role.ASSISTANT, content=ai_answer)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with event-based streaming.

        The Responses API uses a different SSE format than Chat Completions,
        with explicit event types and data fields.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses event-based Server-Sent Events (SSE) format.
            Listens for 'response.output_text.delta' events to build the response.
            Each line with "event: " specifies the event type, followed by "data: " with the payload.
        """
        #TODO:
        # https://developers.openai.com/api/docs/guides/text?lang=curl
        # - Prepare headers with authorization and content type
        # - Prepare input messages
        # - Execute post request to AI API (use `aiohttp`)
        # - Handle stream with events
        # - Parse response
        # - Print chunks to console
        # - Return ASSISTANT message
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        inputs = list(map(lambda ms: ms.to_dict(), messages))

        request_body = json.dumps({
            "model": self._model_name,
            "instructions": self._system_prompt,
            "input": inputs,
            "stream": True
        })

        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=headers, data=request_body) as response:
                response.raise_for_status()
                print("AI:", end="")
                event = None
                ai_content = ''
                async for line in response.content:
                    text_line = line.decode("utf-8").strip()
                    if text_line.startswith("event: "):
                        event = text_line[7:]
                    elif text_line.startswith("data: "):
                        if event == "response.output_text.delta":
                            content_json = json.loads(text_line[6:])
                            content_delta = content_json.get("delta") or ""
                            ai_content += content_delta
                            print(content_delta, flush=True, end="")
                    elif not text_line:
                        event = None

                print()
        return Message(role=Role.ASSISTANT, content=ai_content)
                    
