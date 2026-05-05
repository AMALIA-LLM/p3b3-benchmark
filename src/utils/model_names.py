MODEL_RENAMES = {
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "OLMo-2-1124-7B-Instruct": "OLMo 2-7B",
    "Qwen2.5-7B-Instruct": "Qwen 2.5-7B",
    "gervasio-8b-portuguese-ptpt-decoder": "Gervásio 8B",
    "salamandra-7b-instruct": "Salamandra 7B",
    "gemma-3-12b-it": "Gemma 3 12B",
    'BSC-LT-salamandra-7b-instruct': 'Salamandra 7B',
    'allenai-Olmo-3-7B-Instruct': 'OLMo 3 7B',
    'allenai-Olmo-3.1-32B-Instruct': 'OLMo 3.1 32B',
    'utter-project-EuroLLM-9B-Instruct-2512': 'EuroLLM 9B',
    'utter-project-EuroLLM-22B-Instruct-2512': 'EuroLLM 22B',
    'swiss-ai-Apertus-8B-Instruct-2509': 'Apertus 8B',
    'swiss-ai-Apertus-70B-Instruct-2509': 'Apertus 70B',
    'carminho-AMALIA-9B-50-1225-SFT': 'AMALIA 9B (SFT)',
    'carminho-AMALIA-9B-50-1225-DPO': 'AMALIA 9B',
    "Llama-3.1-8B-Instruct": "LLaMA 3.1 8B",
    'meta-llama-Llama-3.1-8B-Instruct': 'Llama 3.1 8B',
    'meta-llama-Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'PORTULAN-gervasio-8b-portuguese-ptpt-decoder': 'Gervásio 8B',
    'PORTULAN-gervasio-70b-portuguese-ptpt-decoder': 'Gervásio 70B',
    "Ministral-8B-Instruct-2410": "Ministral-8B",
    'mistralai-Ministral-3-14B-Instruct-2512': 'Ministral 3 14B',
    'mistralai-Ministral-3-8B-Instruct-2512': 'Ministral 3 8B',
    'Qwen-Qwen3-8B': 'Qwen 3 8B',
    'Qwen-Qwen3.5-9B': 'Qwen 3.5 9B',
    'Qwen-Qwen3.5-27B': 'Qwen 3.5 27B',
    'Polygl0t-Tucano2-qwen-3.7B-Instruct': 'Tucano 2 3.7B',
    'google-gemma-3-12b-it': 'Gemma 3 12B',
    'google-gemma-3-27b-it': 'Gemma 3 27B',
    'google-gemma-4-E4B-it': 'Gemma 4 E4B',
    'google-gemma-4-31B-it': 'Gemma 4 31B',
    'openai-gpt-oss-20b': 'GPT-OSS 20B',
    'maritaca-api-sabia-3.1': 'Sabiá 3.1',
    'maritaca-api-sabia-4': 'Sabiá 4',
    'google-langchain-api-gemini-3-flash-preview': 'Gemini 3 Flash',
}


def extract_model_name(model_id : str) -> str:
    parts = model_id.split('/')
    if not parts[-1]: parts.pop()
    initial_name = '/'.join(parts[-2:]) if parts[-1].startswith('checkpoint') else parts[-1]
    return MODEL_RENAMES.get(initial_name, initial_name)
