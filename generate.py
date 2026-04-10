#!/usr/bin/env python3

import json, sys, tyro, time, os

from dataclasses import dataclass
from models import HuggingFaceModel, EntityTurn, Gemini, ChatGPT, Model
from typing import TypedDict, cast
from itertools import batched

class Turn(TypedDict):
    user: str
    assistant : str

class Conversation(TypedDict):
    raw_output : str
    turns : list[Turn]


def generate_turns(model : Model, turns : list[list[str]], extra_messages : list[EntityTurn] = []) -> list[Conversation]:
    history         = [cast(list[EntityTurn], []) for _ in turns] # [cast(list[EntityTurn], [])] * len(turns)
    final_raw_input = [''] * len(turns)
    indexes        = list(enumerate(turns))

    for _, msg in indexes: msg.reverse()

    while len(indexes) > 0:

        for i, msgs in indexes:
            history[i].append({ 'role': 'user', 'content': msgs.pop()})

        res = model.generate_with_debug([extra_messages + list(history[i]) for i, _ in indexes], max_connections = 50) # pyright: ignore
        for (i, msgs), output, raw_outputs in zip(indexes, res.outputs, res.raw_outputs):
            history[i].append({
                'role': 'assistant', 'content': output
            })

            if len(msgs) == 0:
                final_raw_input[i] = raw_outputs

        indexes = [entry for entry in indexes if len(entry[1]) > 0]


    assert all(final_raw_input)
    assert all(map(lambda v: len(v) % 2 == 0, history))

    # print(json.dumps(history, indent='  ', ensure_ascii=False))
    # print('-' * 100)
    return [
        {
            'raw_output': raw_input,
            'turns': [{r['role'] : r['content'] for r in roles } for roles in batched(msgs, 2)] # pyright: ignore
        }
        for raw_input, msgs in zip(final_raw_input, history)
    ]

@dataclass
class Config:
    model_name_or_path : str
    base_output_dir : str
    system_prompt : str = ''


def format_path(*path_to_file : str) -> str: 
    """ Preffixes file name with a timestamp and replace all '/' to '-' in the file portion of of the path"""
    time_portion = time.strftime("%Y-%m-%dT%H-%M-%S%z")
    file_name = f'{time_portion}_{path_to_file[-1].replace('/', '-')}' 
    return file_name if len(path_to_file) == 1 else os.path.join(*path_to_file[:-1], file_name)

def load_model(config : Config) -> Model:
    name_or_path = config.model_name_or_path
    api_model_builders = [
        ('google/', Gemini),
        ('openai/', ChatGPT)
    ]

    for prefix, builder in api_model_builders:
        if name_or_path.startswith(prefix):
            model_id = '/'.join( name_or_path.split('/')[1:] )
            print(f"loading API model '{model_id}'")
            return builder(model_id)
    
    from vllm import SamplingParams
    return HuggingFaceModel(
        config.model_name_or_path,
        system_prompt   = config.system_prompt,
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            seed=42,
        )
    )


RENAMES = {
    "47-32k-9B-carminho-with_euroblocks_safety_hermes_customst/checkpoint-2875": "AMALIA-9B 32k v49",
    "47-4k-9B-carminho-with_euroblocks_safety_hermes_customst/checkpoint-13590": "AMALIA-9B 4k v49",
    "47-32k-llama/checkpoint-700": "AMALIA-LLaMA-3.1-8B-32k",
    "49-32k-llama_instruct/checkpoint-1767": "AMALIA-LLaMA-3.1-8B-Instruct-32k",
    "47-32k-qwen3_8B/checkpoint-1482": "AMALIA-Qwen3-8B-32k",
    "49-32k-eurollm-9B/checkpoint-1928": "EuroLLM-AMALIA-9B-32k v49",
    "49-32k-gemma3-12B/checkpoint-1368": "AMALIA-Gemma3-12B-32k",
    "47-safety-dpo-mix_safety_sft_200k/checkpoint-6738_merged": "49 DPO",
    "50-carminho-big/checkpoint-1776": "AMALIA-9B-32k-big v50 (checkpoint 1776)", # 32k SFT-BIG
    "50-carminho-big/checkpoint-3441": "AMALIA-9B-32k-big v50 (checkpoint 3441)", # 32k SFT-BIG
    "50-carminho-big/checkpoint-3480": "AMALIA-9B-32k-big v50 (checkpoint 3480)",
    "50-dpo-mix_safety_sft_200k_if/checkpoint-6892_merged": "AMALIA-9B-32k-big-DPO-small v50",
    "50-carminho-big-old/checkpoint-18501":  "AMALIA-9B-4k-big v50",
    "50-big-4k-dpo-big/checkpoint-6155_merged": "AMALIA-9B-4k-big-DPO-big v50",
    "49-4k-eurollm-9B/checkpoint-12231": "EuroLLM AMALIA-9B 4k v49",
    "Llama-3.1-8B-Instruct": "LLaMA 3.1-Instruct-8B",
    "Ministral-8B-Instruct-2410": "Ministral-8B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "OLMo-2-1124-7B-Instruct": "OLMo 2-7B",
    "Qwen2.5-7B-Instruct": "Qwen 2.5-7B",
    "gervasio-8b-portuguese-ptpt-decoder": "LLaMA-3.1-Gervasio-8B",
    "salamandra-7b-instruct": "Salamandra-7B",
    "gemma-3-12b-it": "Gemma-3-12B",
}


def extract_model_name(model_id : str) -> str:
    parts = model_id.split('/')
    if not parts[-1]: parts.pop()
    initial_name = '/'.join(parts[-2:]) if parts[-1].startswith('checkpoint') else parts[-1]
    return RENAMES.get(initial_name, initial_name)

# def extract_model_id(model_name_or_path : str) -> str:
#     raise Exception()

def do_generation(
        model : Model,
        prompts : list[tuple[int, list[str]]],
        output_dir : str | None, 
        base_output_filename : str,
        id_format : str = '{}',
        extra_messages : list[EntityTurn] = []
    ):

    responses = generate_turns(model, [list(turn) for _, turn in prompts], extra_messages)
    results = [ 
        { 'prompt_id': id_format.format(id), 'model_name': extract_model_name(model.get_name()) } | res 
        for (id, _), res in zip(prompts, responses) 
    ]

    output_file = format_path(
        output_dir or '.',
        base_output_filename        \
                .strip('/')         \
                .lower()            \
                .replace('_', '-')  \
                .replace('/', '_') + '.json'
    )

    with open(output_file, 'w') as file:
        json.dump(results, file, indent='  ', ensure_ascii=False)

    print(f"saved results in '{output_file}")


def main(): 
    args = sys.argv[1:]
    cfg = tyro.cli(Config, args=args if args else ['--help'])

    if not os.path.isdir(cfg.base_output_dir):
        print(f"Error: Output directory '{cfg.base_output_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    with open('./resources/all_prompts.json') as file:
        prompts = list(json.load(file).items())


    model = load_model(cfg)
    # from vllm import SamplingParams
    # model = HuggingFaceModel(
    #     cfg.model_name_or_path,
    #     system_prompt=cfg.system_prompt,
    #     sampling_params = SamplingParams(
    #         temperature=0,
    #         max_tokens=1024,
    #     )
    # )
    #
    do_generation(model, prompts, cfg.base_output_dir, cfg.model_name_or_path)
    do_generation(
        model, prompts, cfg.base_output_dir, cfg.model_name_or_path + '/pt-pt',
        id_format='{}t',
        extra_messages=[
            { 'role': 'user', 'content': 'Responda em português europeu, por favor.' },
            { 'role': 'assistant', 'content': 'Sim, claro.' }
        ]
    )
    do_generation(
        model, prompts, cfg.base_output_dir, cfg.model_name_or_path + '/pt-br',
        id_format='{}r',
        extra_messages=[
            { 'role': 'user', 'content': 'Responda em português brasileiro, por favor.' },
            { 'role': 'assistant', 'content': 'Sim, claro.' }
        ]
    )

if __name__ == '__main__': main()

# srun --pty --gres=gpu:1 --mem 50G  python generate.py --model-name-or-path <model_name> --base-output-dir pt-pt-eval