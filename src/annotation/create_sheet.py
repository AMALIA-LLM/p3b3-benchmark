import os
import json
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pandas as pd


def parse_timestamp_from_filename(filename):
    """Extract timestamp from filename format: 2026-04-24T01-50-11+0100_..."""
    try:
        timestamp_str = filename.split('_')[0]
        # Convert format: 2026-04-24T01-50-11+0100 to 2026-04-24T01:50:11+01:00
        # Replace hyphens in time portion with colons, and format timezone
        date_part, time_and_tz = timestamp_str.split('T')
        if '+' in time_and_tz:
            time_part, tz_part = time_and_tz.split('+')
            time_part = time_part.replace('-', ':')
            # Format timezone: 0100 -> +01:00
            tz_formatted = f"+{tz_part[:2]}:{tz_part[2:]}"
            iso_str = f"{date_part}T{time_part}{tz_formatted}"
        elif '-' in time_and_tz and time_and_tz.count('-') >= 2:
            # Handle negative timezone if present
            parts = time_and_tz.split('-')
            time_part = '-'.join(parts[:2]).replace('-', ':')
            tz_part = parts[-1]
            tz_formatted = f"-{tz_part[:2]}:{tz_part[2:]}"
            iso_str = f"{date_part}T{time_part}{tz_formatted}"
        else:
            # No timezone, just convert time hyphens to colons
            time_part = time_and_tz.replace('-', ':')
            iso_str = f"{date_part}T{time_part}"

        return datetime.fromisoformat(iso_str)
    except Exception as e:
        print(f"Error parsing timestamp from '{filename}': {e}")
        return None


def get_most_recent_file_per_suffix(model_folder):
    """Get the most recent JSON file for each suffix in a model folder."""
    suffix_files = defaultdict(list)

    for filename in os.listdir(model_folder):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(model_folder, filename)

        # Extract timestamp
        timestamp = parse_timestamp_from_filename(filename)
        if not timestamp:
            print("No valid timestamp found in filename:", filename)
            continue

        suffix = "normal"
        if filename.endswith('pt-pt.json'):
            suffix = "pt-pt"
        elif filename.endswith('pt-br.json'):
            suffix = "pt-br"

        suffix_files[suffix].append((timestamp, filepath))

    # Get most recent file for each suffix
    most_recent = {}
    for suffix, files in suffix_files.items():
        if files:
            # Sort by timestamp and get the most recent
            files.sort(reverse=True)
            most_recent[suffix] = files[0][1]

    return most_recent


def select_random_conversations(json_file, num_conversations=2):
    """Select random conversations from a JSON file.

    Returns a list of tuples (conversation, model_name).
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    # Randomly select conversations
    if len(conversations) <= num_conversations:
        selected = conversations
    else:
        selected = random.sample(conversations, num_conversations)

    # Return conversations with their model names
    return [(conv, conv.get('model_name', 'unknown')) for conv in selected]


def create_annotation_sheet(results_dir='results', output_file='outputs/annotation_sheet.csv', num_conversations=2, include_models=None):
    """Create annotation sheet with conversations from all models.

    Args:
        results_dir: Directory containing model result folders
        output_file: Output CSV filename
        num_conversations: Number of random conversations to select per model
        include_models: List of model names to include (default: None, includes all models)
    """
    results_path = Path(results_dir)
    include_models = include_models or []

    # Collect all conversations
    all_rows = []

    # Iterate through model folders
    for model_folder in sorted(results_path.iterdir()):

        if not model_folder.is_dir():
            continue

        model_name = model_folder.name

        # Only consider models in the include list (if provided)
        if include_models and model_name not in include_models:
            print(f"Skipping model: {model_name}")
            continue

        print(f"Processing model: {model_name}")

        # Get most recent file for each suffix
        suffix_files = get_most_recent_file_per_suffix(model_folder)

        for suffix, json_file in suffix_files.items():
            print(f"  - Processing suffix '{suffix}': {Path(json_file).name}")

            # Select random conversations (returns list of tuples)
            conversations_with_models = select_random_conversations(json_file, num_conversations)

            for conv, model_name_from_json in conversations_with_models:
                turns = conv.get('turns', [])

                # Create one row for each turn pair
                for i, turn in enumerate(turns, 1):
                    row = {
                        'conversation_id': conv.get('prompt_id', ''),
                        'model_name': model_name_from_json,
                        'prompt_type': conv.get('prompt_type', suffix),
                        'turn_nr': i,
                        'user': turn.get('user', ''),
                        'assistant': turn.get('assistant', ''),
                        'rating': '',
                        'comment': ''
                    }
                    all_rows.append(row)

    if not all_rows:
        print("No conversations found!")
        return

    # Shuffle all rows to reduce bias
    random.shuffle(all_rows)

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Define column order
    ordered_columns = ['conversation_id', 'model_name', 'prompt_type', 'turn_nr', 'user', 'assistant', 'rating', 'comment']

    # Reorder columns
    df = df[ordered_columns]

    # create folder if not exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\nAnnotation sheet created: {output_file}")
    print(f"Total conversations: {len(df)}")

    # print models considered
    print("\nModels included in annotation sheet:")
    included_models = set(df['model_name'])
    for model in sorted(included_models):
        print(f" - {model}")


def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Example: include these models
    include_models = [
        "amalia-llm-AMALIA-9B-50-1225-DPO",
        "swiss-ai-Apertus-8B-Instruct-2509",
        "utter-project-EuroLLM-9B-Instruct-2512",
        "google-gemma-3-12b-it",
        "meta-llama-Llama-3.1-8B-Instruct",
        "PORTULAN-gervasio-8b-portuguese-ptpt-decoder",
        "mistralai-Ministral-3-14B-Instruct-2512",
        "allenai-Olmo-3-7B-Instruct",
        "Qwen-Qwen3-8B",
        "Qwen-Qwen3.5-9B",
        "BSC-LT-salamandra-7b-instruct",
        "google-gemma-4-E4B-it"
    ]

    create_annotation_sheet(
        results_dir='results',
        output_file='outputs/annotation_sheet.csv',
        num_conversations=2,
        include_models=include_models
    )


if __name__ == '__main__':
    main()
