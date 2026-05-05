import argparse
import datetime
import json
import os.path

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from config.settings import MAX_RETRIES, MAX_CONNECTIONS
from src.evaluation.score_with_llm import build_cumulative_context, parse_llm_evaluation
from src.models.api_models import LangChainGemini
from src.evaluation.prompts import PROMPT_PT_PT_EVAL, PROMPT_PT_PT_EVAL_SIMPLE, PROMPT_EN_EVAL_SIMPLE


def load_conversations_from_results(results_folder: Path) -> Dict[Tuple[str, str, str], List[Dict]]:
    """
    Load all conversations from results folder JSON files.

    Args:
        results_folder: Path to the results folder containing model subfolders with JSON files

    Returns:
        Dictionary mapping (conversation_id, model_name, prompt_type) to turns list
    """
    conversations = {}

    # Iterate through all JSON files in subdirectories
    for json_file in results_folder.glob("*/*.json"):
        # Skip llm_scores and class_scores directories
        if "scores" in json_file.parent.name:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            for conversation in data:
                key = (
                    conversation["prompt_id"],
                    conversation["model_name"],
                    conversation["prompt_type"]
                )
                conversations[key] = conversation["turns"]
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return conversations


def find_conversation_context(
    conversation_id: str,
    model_name: str,
    prompt_type: str,
    turn_nr: int,
    conversations_db: Dict[Tuple[str, str, str], List[Dict]],
    accumulate: bool
) -> Optional[str]:
    """
    Find the conversation context for a specific turn.

    Args:
        conversation_id: Conversation identifier
        model_name: Model name
        prompt_type: Prompt type
        turn_nr: Turn number (1-based)
        conversations_db: Database of all conversations
        accumulate: Whether to accumulate context

    Returns:
        Context string or None if not found
    """
    key = (conversation_id, model_name, prompt_type)

    if key not in conversations_db:
        return None

    turns = conversations_db[key]

    if turn_nr < 1 or turn_nr > len(turns):
        return None

    if accumulate:
        return build_cumulative_context(turns, turn_nr)
    else:
        return turns[turn_nr - 1]['user']


prompt_options = [
    "pt-pt-complete", "pt-pt-complete-no-accumulate",
    "pt-simple", "pt-simple-no-accumulate",
    "en-simple", "en-simple-no-accumulate",
    # etc
]


def main():
    """
    Score conversation responses from annotation sheet using an LLM evaluator.

    Processes annotation sheet CSV, looks up full context from results folder,
    evaluates each turn with cumulative context, and saves results to timestamped CSV files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_sheet", default="outputs/annotation_sheet.csv", type=str, help="Path to the annotation sheet CSV")
    parser.add_argument("--results_folder", default="results", type=str, help="Path to the results folder containing conversation JSON files")
    parser.add_argument("--max_connections", type=int, default=MAX_CONNECTIONS, help="Maximum number of concurrent connections to the LLM API")

    parser.add_argument("--prompt", type=str, choices=prompt_options, default="pt-pt-complete", help="Prompt type")

    # TODO or "gemini-2.5-pro"...
    parser.add_argument("--judge_name", type=str, default='gemini-3-flash-preview', help="LLM judge name for evaluation")
    parser.add_argument("--cache_db_path", type=str, default=".langchain_cache_v2.db", help="Path to cache database")
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES, help="Maximum number of retries for LLM API calls")
    args = parser.parse_args()

    # Load annotation sheet
    print(f"Loading annotation sheet from: {args.annotation_sheet}")
    df = pd.read_csv(args.annotation_sheet)
    print(f"Loaded {len(df)} rows from annotation sheet")

    # Load all conversations from results folder
    print(f"Loading conversations from: {args.results_folder}")
    conversations_db = load_conversations_from_results(Path(args.results_folder))
    print(f"Loaded {len(conversations_db)} conversations from results folder")

    # Initialize LLM client
    gemini_client = LangChainGemini(
        model_name=args.judge_name,
        cache_db_path=args.cache_db_path,
        max_retries=args.max_retries,
    )

    accumulate = False if "no-accumulate" in args.prompt else True

    # Build tasks from annotation sheet
    all_tasks = []
    missing_context_count = 0

    for _, row in df.iterrows():
        conversation_id = row['conversation_id']
        model_name = row['model_name']
        prompt_type = row['prompt_type']
        turn_nr = int(row['turn_nr'])
        user_prompt = row.get('user', '')
        assistant_response = row.get('assistant', '')

        # Look up context from results folder
        context = find_conversation_context(
            conversation_id,
            model_name,
            prompt_type,
            turn_nr,
            conversations_db,
            accumulate=accumulate
        )

        if context is None:
            # Use the user prompt from the sheet as fallback
            context = user_prompt if user_prompt else ""
            missing_context_count += 1
            print(f"Warning: Could not find context for {conversation_id}/{model_name}/{prompt_type}/turn {turn_nr}")

        all_tasks.append((
            model_name,
            conversation_id,
            turn_nr,
            context,
            assistant_response,
            prompt_type
        ))

    if missing_context_count > 0:
        print(f"Warning: Missing context for {missing_context_count}/{len(all_tasks)} entries")

    # Build prompts for evaluation
    if args.prompt == "pt-pt-complete" or args.prompt == "pt-pt-complete-no-accumulate":
        prompts_to_eval = [
            PROMPT_PT_PT_EVAL.format(prompt=task[3], response=task[4])
            for task in all_tasks
        ]
    elif args.prompt == "pt-simple" or args.prompt == "pt-simple-no-accumulate":
        prompts_to_eval = [
            PROMPT_PT_PT_EVAL_SIMPLE.format(prompt=task[3], response=task[4])
            for task in all_tasks
        ]
    elif args.prompt == "en-simple" or args.prompt == "en-simple-no-accumulate":
        prompts_to_eval = [
            PROMPT_EN_EVAL_SIMPLE.format(prompt=task[3], response=task[4])
            for task in all_tasks
        ]
    else:
        raise ValueError(f"Unsupported prompt type: {args.prompt}")

    print(f"Evaluating {len(prompts_to_eval)} prompts from annotation sheet")

    # Generate evaluations
    responses = gemini_client.generate_in_batch(
        prompts=prompts_to_eval,
        max_connections=args.max_connections,
        thinking=False,
    )

    # Process results
    rows = []
    successful_count = 0
    failed_count = 0

    for task, llm_response, raw_prompt in zip(all_tasks, responses, prompts_to_eval):
        model_name, conversation_id, turn_idx, context, response, prompt_type = task
        reasoning, score, not_pt, invalid, success = parse_llm_evaluation(llm_response, conversation_id, turn_idx)

        if success:
            successful_count += 1
        else:
            failed_count += 1

        rows.append({
            'prompt-eval-name': args.prompt,
            'prompt': PROMPT_PT_PT_EVAL,
            'conversation_id': conversation_id,
            'model_name': model_name,
            'prompt_type': prompt_type,
            'turn_nr': turn_idx,
            'accumulate_context': accumulate,
            'context': context,
            'response': response,
            'judge_name': args.judge_name,
            'raw_judge_input': raw_prompt,
            'llm_evaluation_raw': llm_response,
            'reasoning': reasoning,
            'score': score,
            'not_pt': not_pt,
            'invalid': invalid,
        })

    # Save results
    output_df = pd.DataFrame(rows)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S%z")
    annotation_sheet_name = Path(args.annotation_sheet).stem
    output_filename = f"{timestamp}_{annotation_sheet_name}_{args.prompt}_test_judge_scored.csv"
    output_path = os.path.join("outputs/annotations_judge", output_filename)

    output_df.to_csv(output_path, index=False)
    print(f"Saved scored results to: {output_path}")
    print(f"Successfully parsed: {successful_count}/{len(all_tasks)} evaluations")
    print(f"Failed to parse: {failed_count}/{len(all_tasks)} evaluations")


if __name__ == "__main__":
    main()
