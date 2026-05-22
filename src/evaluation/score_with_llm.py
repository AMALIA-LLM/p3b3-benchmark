import argparse
import datetime
import json
import re

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

from config.settings import MAX_RETRIES, MAX_CONNECTIONS
from src.evaluation.llm_judge_init import init_llm_judge_client
from ..evaluation.prompts import PROMPT_PT_PT_EVAL



def build_cumulative_context(conversation_list: List[Dict], turn_idx: int) -> str:
    """
    Build cumulative conversation context up to a specific turn.

    Args:
        conversation_list: List of conversation turns with 'user' and 'assistant' keys
        turn_idx: Current turn index (1-based)

    Returns:
        Formatted string with all previous turns and current user message
    """
    context_parts = []
    for prev_turn in conversation_list[:turn_idx - 1]:
        context_parts.append(f"User: {prev_turn['user']}")
        context_parts.append(f"Assistant: {prev_turn['assistant']}")
    context_parts.append(f"User: {conversation_list[turn_idx - 1]['user']}")
    return "\n".join(context_parts)


def process_conversation(conversation_list: List[Dict], model_name: str, conversation_id: str,
                         prompt_type: str, accumulate: bool) -> List[Tuple]:
    """
    Create evaluation tasks for each turn in a conversation.

    Args:
        conversation_list: List of conversation turns
        model_name: Name of the model being evaluated
        conversation_id: Unique identifier for the conversation
        prompt_type: Type of prompt used
        accumulate: Accumulate evaluation results

    Returns:
        List of tuples containing (model_name, conversation_id, turn_idx, context, response, prompt_type)
    """
    tasks = []
    if accumulate:
        # accumulate context
        for turn_idx, turn in enumerate(conversation_list, start=1):
            context = build_cumulative_context(conversation_list, turn_idx)
            response = turn['assistant']
            tasks.append((model_name, conversation_id, turn_idx, context, response, prompt_type))
    else:
        # only current turn as context
        for turn_idx, turn in enumerate(conversation_list, start=1):
            context = turn['user']
            response = turn['assistant']
            tasks.append((model_name, conversation_id, turn_idx, context, response, prompt_type))
    return tasks


def parse_llm_evaluation(llm_response: str, conversation_id: str, turn_idx: int) -> Tuple[str, int, bool, bool, bool]:
    """
    Parse LLM evaluation response from JSON format.

    Args:
        llm_response: Raw JSON string response from LLM
        conversation_id: Conversation identifier for error reporting
        turn_idx: Turn index for error reporting

    Returns:
        Tuple of (reasoning, score, not_pt, invalid, success) where success is True if parsing succeeded
    """
    try:
        # Extract JSON from response (because we asked for a json format response)
        json_match = re.search(r'```json\s*({.*?})\s*```', llm_response, re.DOTALL)
        if json_match:
            result = json_match.group(1).strip()
        else:
            json_match = re.search(r'({\s*".*?})', llm_response, re.DOTALL)
            if json_match:
                result = json_match.group(1).strip()
            else:
                return None, None, None, None, False

        parsed = json.loads(result)

        return parsed["reasoning"], parsed["score"], parsed["not_pt"], parsed["invalid"], True
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Warning: Invalid LLM response for conversation {conversation_id} turn {turn_idx}: {e}")
        return None, None, None, None, False


def main():
    """
    Score conversation responses using an LLM evaluator.

    Processes JSON files containing conversations, evaluates each turn with cumulative context,
    and saves results to timestamped CSV files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Path to the folder containing the responses to be scored")
    parser.add_argument("--max_connections", type=int, default=MAX_CONNECTIONS, help="Maximum number of concurrent connections to the LLM API")
    parser.add_argument("--judge_name", type=str, default='gemini-3-flash-preview', help="LLM judge name for evaluation")
    parser.add_argument("--no-accumulate-context", action="store_true", help="Only use current turn as context instead of accumulating previous turns")
    parser.add_argument("--cache_db_path", type=str, default=".langchain_cache_v2.db", help="Path to cache database")
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES, help="Maximum number of retries for LLM API calls")
    parser.add_argument("--sub_folder_name", type=str, default="llm_scores", help="Name of the subfolder to save scored results (default: llm_scores)")
    args = parser.parse_args()

    # Initialize LLM client
    judge_client = init_llm_judge_client(
        judge_name=args.judge_name,
        cache_db_path=args.cache_db_path,
        max_retries=args.max_retries,
    )

    # iterate over all outputs for a model
    for json_file in Path(args.input_folder).glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        all_tasks = []
        # iterate over all conversations
        for conversation in data:
            tasks = process_conversation(
                conversation_list=conversation["turns"],
                model_name=conversation["model_name"],
                conversation_id=conversation["prompt_id"],
                prompt_type=conversation["prompt_type"],
                accumulate=not args.no_accumulate_context,
            )
            all_tasks.extend(tasks)

        # create the prompts
        prompts_to_eval = [
            PROMPT_PT_PT_EVAL.format(prompt=task[3], response=task[4])
            for task in all_tasks
        ]

        # generate judge scores
        print(f"Evaluating {len(prompts_to_eval)} prompts from file: {json_file.name}")

        responses = judge_client.generate_in_batch(
            prompts=prompts_to_eval,
            max_connections=args.max_connections,
            thinking=False,
        )

        output_dir = Path(args.input_folder) / args.sub_folder_name
        output_dir.mkdir(exist_ok=True)

        rows = []
        successful_count = 0
        failed_count = 0

        # parse the outputs
        for task, llm_response, raw_prompt in zip(all_tasks, responses, prompts_to_eval):
            model_name, conversation_id, turn_idx, context, response, prompt_type = task
            reasoning, score, not_pt, invalid, success = parse_llm_evaluation(llm_response, conversation_id, turn_idx)

            if success:
                successful_count += 1
            else:
                failed_count += 1

            rows.append({
                'conversation_id': conversation_id,
                'model_name': model_name,
                'prompt_type': prompt_type,
                'turn_nr': turn_idx,
                'accumulate_context': not args.no_accumulate_context,
                'context': context,
                'response': response,
                'judge_name': args.judge_name,
                'raw_prompt': raw_prompt,
                'llm_evaluation_raw': llm_response,
                'reasoning': reasoning,
                'score': score,
                'not_pt': not_pt,
                'invalid': invalid,
            })

        # save the results
        df = pd.DataFrame(rows)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S%z")
        output_filename = f"{timestamp}_{json_file.stem}_scored_v1.csv"
        output_path = output_dir / output_filename

        df.to_csv(output_path, index=False)
        print(f"Saved scored results to: {output_path}")
        print(f"Successfully parsed: {successful_count}/{len(all_tasks)} evaluations")
        print(f"Failed to parse: {failed_count}/{len(all_tasks)} evaluations")


if __name__ == "__main__":
    main()
