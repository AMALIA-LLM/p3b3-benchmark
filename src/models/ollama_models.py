import os

from dotenv import load_dotenv
from ollama import Client as OllamaClient

from config.settings import MAX_OUTPUT_TOKENS
from .base import Model, BatchedPrompts
from .utils import parallel_generation


# load env variables
load_dotenv()


class Ollama(Model):
    """
    Ollama model implementation for locally hosted models.

    By default, connects to http://localhost:11434, but can be configured via OLLAMA_HOST env var.
    """

    def __init__(self, model_name: str, host: str | None = None, api_key: str | None = None):
        """
        Initialize Ollama model.

        Args:
            model_name: Name of the Ollama model (e.g., 'llama3.2', 'mistral', 'codellama')
            host: Optional Ollama server host URL (defaults to localhost:11434 or OLLAMA_HOST env var)
        """
        self.model_name = model_name
        ollama_host = host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.client = OllamaClient(host=ollama_host, api_key=api_key)

    def get_name(self) -> str:
        return self.model_name

    def generate_in_batch(self, prompts: BatchedPrompts, max_connections: int, thinking: bool = False) -> list[str]:
        """
        Generate responses for a batch of prompts using Ollama.

        Args:
            prompts: Either list of strings or list of message histories
            max_connections: Maximum parallel connections
            thinking: Whether or not to use thinking mode

        Returns:
            List of generated responses
        """
        assert len(prompts), 'prompts list should not be empty'

        # Convert string prompts to message format
        batched_messages = prompts if isinstance(prompts[0], list) else \
            [[{'role': 'user', 'content': prompt}] for prompt in prompts]

        def generate_response(messages: list[dict]) -> str:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,  # type: ignore
                options={
                    'temperature': 0,
                    'seed': 42,
                    'max_output_tokens': MAX_OUTPUT_TOKENS,
                },
                think=thinking
            )
            return response['message']['content']  # type: ignore

        return parallel_generation(
            generate_response,
            batched_messages,
            max_connections
        )
