from dataclasses import dataclass
from typing import TypedDict, Literal


@dataclass
class GenerationResult:
    outputs: list[str]
    raw_outputs: list[str]


class EntityTurn(TypedDict):
    role: Literal['user', 'assistant', 'system']
    content: str


TurnedPrompt = list[EntityTurn]
BatchedPrompts = list[str] | list[TurnedPrompt]


class ResponseIsNone(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'model response is none'


class Model:
    def get_name(self) -> str: ...
    def generate_in_batch(self, prompts: BatchedPrompts, max_connections: int, thinking: bool = False) -> list[str]: ...
    def generate(self, prompts: BatchedPrompts, thinking: bool = False) -> list[str]:
        return self.generate_in_batch(prompts, max_connections=10, thinking=thinking)
    def chat(self, messages, thinking: bool = False) -> str:
        return self.generate([messages], thinking=thinking)[0]
    def generate_with_debug(self, prompts: BatchedPrompts, max_connections: int, thinking: bool = False) -> GenerationResult:
        outputs = self.generate_in_batch(prompts, max_connections, thinking=thinking)  # pyright: ignore
        return GenerationResult(
            outputs, raw_outputs=[
               (
                   f'user:\n{ctx}' if isinstance(ctx, str) else
                       '\n\n'.join(f'{p['role']}:\n{p['content']}' for p in ctx)
               ) + '\n\nassistant:\n' + output  for ctx, output in zip(prompts, outputs)
            ]
        )
