import torch
from vllm import LLM
from .base import Model, BatchedPrompts, TurnedPrompt, GenerationResult
from config.settings import MAX_MODEL_LEN


class VLLMModel(Model):

    def __init__(self, model_name: str, system_prompt: str | None = None, **gen_args):
        self.model_name = model_name
        self.num_gpus = torch.cuda.device_count()
        print(f'Using {self.num_gpus} GPUs')

        # special case for tucano
        max_size = MAX_MODEL_LEN
        if "tucano" in model_name.lower():
            max_size = 4096
        elif "salamandra" in model_name.lower():
            max_size = 8192

        self.model = LLM(model=model_name, seed=42, dtype="bfloat16", max_model_len=max_size,
                         gpu_memory_utilization=0.80,  # TODO change this as needed
                         tensor_parallel_size=self.num_gpus)
        self.extra_msgs: TurnedPrompt = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.get_args = gen_args

    def get_name(self) -> str:
        return self.model_name

    def generate_with_debug(self, prompts: BatchedPrompts, max_connections: int, thinking: bool = False) -> GenerationResult:
        assert len(prompts) > 0, "list of prompts should not be empty"

        chats = prompts if isinstance(prompts[0], list) else \
            [[{"role": 'user', 'content': prompt}] for prompt in prompts]

        # Prepend extra messages (like system prompt) to each conversation
        if self.extra_msgs:
            full_conversations = [self.extra_msgs + chat for chat in chats]
        else:
            full_conversations = [chat for chat in chats]

        # Use vLLM's built-in chat format handling
        results = self.model.chat(
            messages=full_conversations,
            chat_template_kwargs={"enable_thinking": thinking},
            **self.get_args
        )

        return GenerationResult(
            outputs=[v.outputs[0].text for v in results],
            raw_outputs=[str(v) for v in results]
        )

    def generate_in_batch(self, prompts: BatchedPrompts, max_connections: int, thinking: bool = False) -> list[str]:
        return self.generate_with_debug(prompts, max_connections, thinking).outputs
