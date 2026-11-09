from google import genai
from google.genai import types

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class GeminiAIClient(AIClient):
    """
    Client for Google Gemini API using the official SDK.

    This implementation uses the official Google GenAI Python library to interact
    with Gemini models, providing both synchronous and streaming response capabilities.

    Attributes:
        _client (genai.Client): Google GenAI client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Gemini client with SDK.

        Args:
            endpoint (str): The Gemini API endpoint (for compatibility, not used by SDK).
            model_name (str): The Gemini model to use (e.g., 'gemini-3-flash-preview').
            api_key (str): The Google API key for authentication.
            system_prompt (str): The system instruction to guide the model's behavior.
        """
        #TODO:
        # Call to __init__ of super class
        # Add genai.Client https://ai.google.dev/gemini-api/docs/text-generation#python_4
        super().__init__(endpoint, model_name, api_key, system_prompt)
        self.client = genai.Client(api_key=api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from Google's Gemini API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Note:
            Gemini uses 'system_instruction' parameter for system-level guidance.
            The response is printed to stdout before being returned.
        """
        #TODO:
        # - Add System prompt
        # - Call client
        # - Print response to console
        # - Return ASSISTANT message
        inputs = self._convert_msg(messages)
        interaction = self.client.interactions.create(
            model=self._model_name,
            system_instruction=self._system_prompt,
            input=inputs,
            store=False
        )
        print("AI:", interaction.output_text)
        return Message(Role.ASSISTANT, interaction.output_text)

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

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from Google's Gemini API.

        The response is streamed chunk-by-chunk, with each text chunk printed
        immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            Uses the async streaming interface provided by the Gemini SDK.
            Each chunk's text is printed to stdout as it arrives.
        """
        #TODO:
        # - Add System prompt
        # - Call client with streaming mode
        # - Handle stream with chunks
        # - Print response to console
        # - Return ASSISTANT message
        inputs = self._convert_msg(messages)
        stream = await self.client.aio.interactions.create(
            model=self._model_name,
            input=inputs,
            stream=True,
            store=False
            )
        ai_content = ""
        print("AI:", end="")
        async for event in stream:
            if event.event_type == "step.delta":
                if event.delta.type == "text":
                    ai_content += event.delta.text or ""
                    print(event.delta.text, flush=True, end="")

        print()
        return Message(role=Role.ASSISTANT, content=ai_content)