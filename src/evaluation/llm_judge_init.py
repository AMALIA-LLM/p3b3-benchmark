from vllm import SamplingParams

from config.settings import MAX_OUTPUT_TOKENS
from src.models.vllm_models import VLLMModel
from src.models.api_models import OpenAICompatible, LangChainGemini
from src.models.ollama_models import Ollama


def init_llm_judge_client(judge_name: str,
                          cache_db_path: str,
                          max_retries: int):

    if "gemini" in judge_name:
        judge_client = LangChainGemini(
            model_name=judge_name,
            cache_db_path=cache_db_path,
            max_retries=max_retries,
        )
    elif "ollama" in judge_name:
        # Support Ollama models (e.g., "ollama/llama3.2")
        model_name = judge_name.replace("ollama/", "")
        judge_client = Ollama(model_name=model_name)
    elif "openai-compatible" in judge_name:
        # Support OpenAI compatible models - e.g. vllm server
        model_name = judge_name.replace("openai-compatible/", "")
        judge_client = OpenAICompatible(model_name=model_name)
    elif "vllm" in judge_name:
        model_name = judge_name.replace("vllm/", "")
        judge_client = VLLMModel(
            model_name=model_name,
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                seed=42,
            )
        )
    else:
        # TODO add other judge options
        raise Exception(f"Unsupported judge model: {judge_name}. Use 'gemini-*' or 'ollama/*' or 'openai-compatible/*'")

    return judge_client