import json, sys, tyro, time, os
from dataclasses import dataclass
from src.models.base import EntityTurn, Model
from src.models.api_models import LangChainGemini, Sabia
from src.models.ollama_models import Ollama
from src.models.vllm_models import VLLMModel
from config.settings import MAX_OUTPUT_TOKENS, MAX_CONNECTIONS
from typing import TypedDict, Optional, List, Tuple
from vllm import SamplingParams

from src.utils.model_names import extract_model_name


class Turn(TypedDict):
    user: str
    assistant : str


class Conversation(TypedDict):
    raw_output : str
    turns : list[Turn]


def generate_turns(model : Model,
                   max_connections: int,
                   turns: List[List[str]],
                   extra_messages : Optional[List[EntityTurn]] = None,
                   ) -> List[Conversation]:
    """
    Generate multi-turn conversations by alternating between user and assistant messages.

    Args:
        model: The language model to use for generation
        max_connections: Maximum number of connections to generate
        turns: List of conversation turns, where each conversation is a list of user messages
        extra_messages: Optional prefix messages to prepend to each conversation history

    Returns:
        List of Conversation objects containing the raw model output and structured turns

    Note:
        This function processes conversations in parallel, handling different conversation
        lengths by filtering out completed conversations after each generation round.
    """
    if not turns:
        return []

    # Check if we need to concatenate the last extra_message with first turn
    should_concatenate = (
        extra_messages is not None and
        len(extra_messages) > 0 and
        extra_messages[-1]['role'] == 'user'
    )

    # Initialize storage for each conversation
    num_convs = len(turns)
    last_extra_user_content: Optional[str] = None
    base_messages: Optional[List[EntityTurn]] = None

    if should_concatenate and extra_messages is not None:
        # Remove the last user message from extra_messages to concatenate later
        base_messages = extra_messages[:-1]
        last_extra_user_content = extra_messages[-1]['content']
    else:
        base_messages = extra_messages

    histories: List[List[EntityTurn]] = [list(base_messages) if base_messages else [] for _ in range(num_convs)]
    completed_turns: List[List[Turn]] = [[] for _ in range(num_convs)]
    current_turn_idx: List[int] = [0] * num_convs

    # Track which conversations are still active (have more turns to process)
    active_indices = list(range(num_convs))

    # Process turns until all conversations are complete
    while active_indices:
        # Prepare batch of prompts for the current turn
        batch_prompts: List[List[EntityTurn]] = []
        batch_conv_indices: List[int] = []

        for idx in active_indices:
            turn_idx = current_turn_idx[idx]
            if turn_idx < len(turns[idx]):
                # Add user message to history
                user_msg = turns[idx][turn_idx]

                # Concatenate with last extra_message if it's the first turn and needed
                if turn_idx == 0 and last_extra_user_content:
                    user_msg = last_extra_user_content + '\n' + user_msg

                histories[idx].append(EntityTurn(role='user', content=user_msg))

                # Add to batch
                batch_prompts.append(histories[idx].copy())
                batch_conv_indices.append(idx)

        if not batch_prompts:
            break

        # special case for sabia due to rate limits being lower
        if isinstance(model, Sabia):
            max_connections = 1

        # Generate responses in batch
        responses = model.generate_in_batch(batch_prompts, max_connections=max_connections)

        # Update conversation states with responses
        for conv_idx, response in zip(batch_conv_indices, responses):
            turn_idx = current_turn_idx[conv_idx]

            # Add assistant response to history
            histories[conv_idx].append(EntityTurn(role='assistant', content=response))

            # Store the completed turn
            completed_turns[conv_idx].append(Turn(
                user=turns[conv_idx][turn_idx],
                assistant=response
            ))

            # Move to next turn
            current_turn_idx[conv_idx] += 1

        # Filter out completed conversations
        active_indices = [
            idx for idx in active_indices
            if current_turn_idx[idx] < len(turns[idx])
        ]

    # Format results
    results: List[Conversation] = []
    for i in range(num_convs):
        # Create raw output by concatenating all messages
        raw_messages = [f"{msg['role']}: {msg['content']}" for msg in histories[i]]
        raw_output = '\n\n'.join(raw_messages)

        results.append(Conversation(
            raw_output=raw_output,
            turns=completed_turns[i]
        ))

    return results



@dataclass
class Config:
    model_name_or_path : str
    base_output_dir : str = ""
    dataset_path : str = 'resources/all_prompts.json'  # Path to the dataset file
    generations : str = 'all'  # Options: 'all', 'no-bias', 'pt-pt', 'pt-br', or comma-separated (e.g., 'no-bias,pt-pt')
    max_connections : int = MAX_CONNECTIONS

def load_model(config : Config) -> Model:
    """
    Load a language model based on the configuration.

    Args:
        config: Configuration object containing model name/path and settings

    Returns:
        Model instance (either API-based or HuggingFace local model)

    Note:
        Models with prefix 'google-langchain-api/' load Gemini, 'ollama/' load Ollama, 'sabia/' load Sabia
        All other paths are treated as HuggingFace models loaded locally with vLLM (GPU needed).
    """
    name_or_path = config.model_name_or_path
    api_model_builders = [
        ('google-langchain-api/', LangChainGemini),
        ('ollama/', Ollama),
        ('maritaca-api/', Sabia),
    ]

    for prefix, builder in api_model_builders:
        if name_or_path.startswith(prefix):
            model_id = '/'.join( name_or_path.split('/')[1:] )
            print(f"loading API model '{model_id}'")
            return builder(model_id)

    return VLLMModel(
        config.model_name_or_path,
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            seed=42,
        )
    )


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name
    """
    return model_name.strip().replace(' ', '-').replace('_', '-').replace('/', '-').lower()


def do_generation(
        model : Model,
        max_connections : int,
        prompts : List[Tuple[int, List[str]]],
        output_dir : str | None,
        base_output_filename : str,
        prompt_type : str,
        dataset_name : str,
        id_format : str = '{}',
        extra_messages : Optional[List[EntityTurn]] = None
    ):
    """
    Generate conversations and save results to a timestamped JSON file.

    Args:
        model: The language model to use for generation
        max_connections: Maximum number of connections to use for generation
        prompts: List of (id, turns) tuples where id is the prompt identifier
        output_dir: Directory to save output file (uses current dir if None)
        base_output_filename: Base name for the output file (will be processed and timestamped)
        prompt_type: Type of prompt (e.g., "normal", "pt-pt", "pt-br")
        dataset_name: Name of the dataset being used
        id_format: Format string for prompt IDs (default: '{}')
        extra_messages: Additional messages to prepend to each conversation

    Side effects:
        Writes a JSON file with conversation results to the output directory
    """

    responses = generate_turns(model, max_connections, [list(turn) for _, turn in prompts], extra_messages)
    results = [
        { 'prompt_id': id_format.format(p_id), 'model_name': extract_model_name(model.get_name()), 'prompt_type': prompt_type, 'dataset': dataset_name} | res
        for (p_id, _), res in zip(prompts, responses)
    ]

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S%z")
    filename = sanitize_model_name(f"{dataset_name}_{base_output_filename}") + '.json'
    output_file = os.path.join(output_dir or 'results', f'{timestamp}_{filename}')

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"saved results in '{output_file}'")


def main():
    """
    Main entry point for the generation script.

    Loads prompts from the specified dataset and generates three sets of responses:
    1. No bias (standard prompts)
    2. pt-PT bias (Portuguese European)
    3. pt-BR bias (Brazilian Portuguese)

    Command line arguments are parsed via tyro from the Config dataclass.
    """
    print("Starting...")

    args = sys.argv[1:]
    cfg = tyro.cli(Config, args=args if args else ['--help'])

    if not cfg.base_output_dir:
        cfg.base_output_dir = "results/" + cfg.model_name_or_path.replace('/', '-')

    if not os.path.isdir(cfg.base_output_dir):
        print(f"Directory '{cfg.base_output_dir}' does not exist. Creating")
        os.makedirs(cfg.base_output_dir, exist_ok=True)

    # Extract dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(cfg.dataset_path))[0]
    print(f"Loading dataset: {dataset_name}")

    # load the data
    with open(cfg.dataset_path) as file:
        prompts = list(json.load(file).items())

    # load the model
    model = load_model(cfg)

    # Parse which generations to run
    gen_options = cfg.generations.lower().replace(' ', '').split(',')
    run_all = 'all' in gen_options
    run_no_bias = run_all or 'no-bias' in gen_options
    run_pt_pt = run_all or 'pt-pt' in gen_options
    run_pt_br = run_all or 'pt-br' in gen_options

    # no bias generation
    if run_no_bias:
        print("Running no-bias generation...")
        do_generation(
            model=model,
            max_connections=cfg.max_connections,
            prompts=prompts,
            output_dir=cfg.base_output_dir,
            base_output_filename=cfg.model_name_or_path,
            prompt_type="normal",
            dataset_name=dataset_name
        )

    # pt-PT bias generation
    if run_pt_pt:
        print("Running pt-PT generation...")
        do_generation(
            model=model,
            max_connections=cfg.max_connections,
            prompts=prompts,
            output_dir=cfg.base_output_dir,
            base_output_filename=cfg.model_name_or_path + '/pt-pt',
            prompt_type="pt-pt",
            dataset_name=dataset_name,
            id_format='{}t',
            extra_messages=[
                EntityTurn(role='user', content='Responde sempre em português europeu (pt-PT).'),
            ]
        )

    # pt-BR bias generation
    if run_pt_br:
        print("Running pt-BR generation...")
        do_generation(
            model=model,
            max_connections=cfg.max_connections,
            prompts=prompts,
            output_dir=cfg.base_output_dir,
            base_output_filename=cfg.model_name_or_path + '/pt-br',
            prompt_type="pt-br",
            dataset_name=dataset_name,
            id_format='{}r',
            extra_messages=[
                EntityTurn(role='user', content='Responde sempre em português brasileiro (pt-BR).'),
            ]
        )

    print("Done.")


if __name__ == '__main__':
    main()
