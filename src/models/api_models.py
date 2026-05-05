import os
from typing import Union
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from .base import Model, BatchedPrompts
from .utils import parallel_generation
from config.settings import MAX_OUTPUT_TOKENS, MAX_RETRIES

# load env variables
load_dotenv()


class LangChainGemini(Model):
    """
    LangChain implementation of Gemini with SQLite caching and built-in retries.

    Features:
    - SQLite cache to avoid redundant API calls
    - Automatic retries with exponential backoff
    - Compatible with existing Model interface
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        cache_db_path: str = ".langchain_cache.db",
        max_retries: int = MAX_RETRIES
    ):
        """
        Initialize LangChain Gemini model with caching and retries.

        Args:
            model_name: Gemini model name (default: gemini-2.0-flash-exp)
            cache_db_path: Path to SQLite cache database (default: .langchain_cache.db)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.model_name = model_name

        # Set up SQLite cache for LangChain
        set_llm_cache(SQLiteCache(database_path=cache_db_path))

        # Initialize the LangChain ChatGoogleGenerativeAI with retries
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=max_retries,
            google_api_key=os.getenv('GEMINI_API_KEY'),
            max_output_tokens=10000,
        )

    def get_name(self) -> str:
        return f"langchain_{self.model_name}"

    def _format_messages(self, prompt: Union[str, list[dict]]) -> list[tuple[str, str]]:
        """
        Convert prompts to LangChain message format.

        Args:
            prompt: Either a string or list of message dicts

        Returns:
            List of (role, content) tuples for LangChain
        """
        if isinstance(prompt, str):
            return [("user", prompt)]

        formatted = []
        for msg in prompt:
            role = msg['role']
            # Map roles to LangChain format
            if role == 'system':
                formatted.append(("system", msg['content']))
            elif role == 'user':
                formatted.append(("user", msg['content']))
            elif role == 'assistant':
                formatted.append(("assistant", msg['content']))

        return formatted

    def generate_in_batch(
        self,
        prompts: BatchedPrompts,
        max_connections: int,
        thinking: bool = False
    ) -> list[str]:
        """
        Generate responses for a batch of prompts using LangChain Gemini.

        Processes requests in chunks with a progress bar showing real-time progress.

        Args:
            prompts: Either list of strings or list of message histories
            max_connections: Maximum parallel connections (processes this many at a time)
            thinking: Whether to enable thinking mode (not used in this implementation)

        Returns:
            List of generated responses
        """
        assert len(prompts), 'prompts list should not be empty'

        # Convert all prompts to LangChain message format
        batch_messages = [self._format_messages(prompt) for prompt in prompts]

        final_responses = []

        # Process in chunks with progress bar
        with tqdm(total=len(batch_messages), desc="Generating responses") as pbar:
            for i in range(0, len(batch_messages), max_connections):
                chunk = batch_messages[i:i + max_connections]

                # Process this chunk
                responses = self.llm.batch(
                    chunk,
                    config={"max_concurrency": max_connections},
                    return_exceptions=True,
                )

                # Process responses from this chunk
                for j, res in enumerate(responses):
                    if isinstance(res, Exception):
                        print(f"Prompt {i+j} failed with error: {res}")
                        final_responses.append("")
                    else:

                        # print only first and last
                        if j == 0 or j == len(responses) - 1:
                            print("=====")
                            print(res.content)
                            print("=====")

                        if isinstance(res.content, list):
                            # grab the text
                            final_responses.append(res.content[-1].get("text", ""))
                        else:
                            final_responses.append(res.content)

                # Update progress bar by the number of items processed in this chunk
                pbar.update(len(chunk))

        return final_responses


class Sabia(Model):
    """
    Sabia model from Maritaca AI using OpenAI-compatible API.

    Uses the OpenAI SDK but with a custom base URL pointing to Maritaca's API.
    Note: The API uses 'responses.create' instead of 'chat.completions.create'
    and returns results in a different format.
    """

    def __init__(self, model_name: str = "sabia-4"):
        """
        Initialize Sabia model.

        Args:
            model_name: Name of the Sabia model (default: sabia-4)
        """
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv('MARITACA_API_KEY'),
            base_url="https://chat.maritaca.ai/api"
        )

    def get_name(self) -> str:
        return self.model_name

    def generate_in_batch(self, prompts: BatchedPrompts, max_connections: int, thinking: bool = False) -> list[str]:
        """
        Generate responses for a batch of prompts using Sabia.

        Args:
            prompts: Either list of strings or list of message histories
            max_connections: Maximum parallel connections
            thinking: Whether to enable thinking mode (not used in this implementation)

        Returns:
            List of generated responses
        """
        assert len(prompts), 'prompts list should not be empty'

        # Convert string prompts to message format
        batched_messages = prompts if isinstance(prompts[0], list) else \
            [[{'role': 'user', 'content': prompt}] for prompt in prompts]

        def generate_response(messages: Union[str, list[dict]]) -> str:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )
            # Extract text from the response format
            return response.output[0].content[0].text

        return parallel_generation(
            generate_response,
            batched_messages,
            max_connections
        )
