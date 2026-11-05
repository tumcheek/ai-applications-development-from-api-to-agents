from anthropic import Anthropic, AsyncAnthropic

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class AnthropicAIClient(AIClient):
    """
    Client for Anthropic's Claude API using the official SDK.

    This implementation uses the official Anthropic Python library to interact
    with Claude models, providing both synchronous and streaming response capabilities.

    Attributes:
        _client (Anthropic): Synchronous Anthropic client instance.
        _async_client (AsyncAnthropic): Asynchronous Anthropic client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Anthropic client with SDK.

        Args:
            endpoint (str): The Anthropic API endpoint (for compatibility, not used by SDK).
            model_name (str): The Claude model to use (e.g., 'claude-sonnet-5', 'claude-haiku-4-5').
            api_key (str): The Anthropic API key for authentication.
            system_prompt (str): The system instruction to guide Claude's behavior.
        """
        #TODO:
        # Call to __init__ of super class
        # Add Anthropic and AsyncAnthropic clients https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#usage
        # (In readme you can find samples with both of these clients)
        # Useful links with request/response samples:
        #   - https://docs.anthropic.com/en/api/overview
        #   - https://docs.anthropic.com/en/api/messages
        super().__init__(endpoint, model_name, api_key, system_prompt)
        self.client = Anthropic(api_key=self._api_key)
        self.async_client = AsyncAnthropic(api_key=self._api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from Anthropic's Claude API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Note:
            Claude's API uses a separate 'system' parameter for system instructions.
            Response content blocks are concatenated into a single text response.
            The response is printed to stdout before being returned.
        """
        #TODO:
        # - Add System prompt
        # - Call client
        # - Print response to console
        # - Return ASSISTANT message
        messages = list(map(lambda ms: ms.to_dict(), messages))
        message = self.client.messages.create(
             max_tokens=1024,
             messages=messages,
             system=self._system_prompt,
             model=self._model_name
        )
        ai_answer = ""
        for content in message.content:
            if content.type == "text":
                ai_answer = content.text
        print(f"AI: {ai_answer}")
        return Message(role=Role.ASSISTANT, content=ai_answer)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from Anthropic's Claude API.

        The response is streamed using event-based streaming, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Listens for 'content_block_delta' events with text deltas.
            Each delta is printed to stdout as it arrives for real-time display.
        """
        #TODO:
        # - Add System prompt
        # - Call client with streaming mode
        # - Handle stream with chunks
        # - Print response to console
        # - Return ASSISTANT message
        messages = list(map(lambda ms: ms.to_dict(), messages))
        stream = await self.async_client.messages.create(
             max_tokens=1024,
             messages=messages,
             system=self._system_prompt,
             model=self._model_name,
             stream=True
        )
        ai_content = ''
        print("AI:", end="")
        async for event in stream:
            if event.type == "content_block_delta":
                delta_text = event.delta.text if event.delta.type == "text_delta" else ""
                ai_content += delta_text
                print(delta_text, end="", flush=True)
        print()

        return Message(role=Role.ASSISTANT, content=ai_content)

