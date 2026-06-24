from collections import defaultdict
from datetime import datetime


def get_model_sort_key():
    """
    Define the preferred model order. Models not in this list will be placed at the end.
    """
    model_order = [
        'BSC-LT-salamandra-7b-instruct',
        'allenai-Olmo-3-7B-Instruct',
        'allenai-Olmo-3.1-32B-Instruct',
        'utter-project-EuroLLM-9B-Instruct-2512',
        'utter-project-EuroLLM-22B-Instruct-2512',
        'swiss-ai-Apertus-8B-Instruct-2509',
        'swiss-ai-Apertus-70B-Instruct-2509',
        'amalia-llm-AMALIA-9B-50-1225-SFT',
        'amalia-llm-AMALIA-9B-50-1225-DPO',
        'meta-llama-Llama-3.1-8B-Instruct',
        'meta-llama-Llama-3.3-70B-Instruct',
        'PORTULAN-gervasio-8b-portuguese-ptpt-decoder',
        'PORTULAN-gervasio-70b-portuguese-ptpt-decoder',
        'mistralai-Ministral-3-14B-Instruct-2512',
        'mistralai-Ministral-3-8B-Instruct-2512',
        'Qwen-Qwen3-8B',
        'Qwen-Qwen3.5-9B',
        'Qwen-Qwen3.5-27B',
        'Polygl0t-Tucano2-qwen-3.7B-Instruct',
        'google-gemma-3-12b-it',
        'google-gemma-3-27b-it',
        'google-gemma-4-E4B-it',
        'google-gemma-4-31B-it',
        'openai-gpt-oss-20b',
        'maritaca-api-sabia-3.1',
        'maritaca-api-sabia-4',
        'google-langchain-api-gemini-3-flash-preview',
    ]

    def sort_key(model_name):
        """Return a tuple for sorting: (is_in_order_list, position_or_name)"""
        if model_name in model_order:
            return 0, model_order.index(model_name)
        else:
            return 1, model_name

    return sort_key


def sort_dataframe_by_model_order(df):
    """
    Sort a DataFrame by the custom model order.
    Works for both DataFrames with 'model' as index or as a column.
    """
    sort_key = get_model_sort_key()

    if 'model' in df.index.names:
        # Model is the index
        sorted_index = sorted(df.index, key=sort_key)
        return df.loc[sorted_index]
    elif 'model' in df.columns:
        # Model is a column
        df_copy = df.copy()
        df_copy['_sort_key'] = df_copy['model'].apply(lambda x: sort_key(x))
        df_copy = df_copy.sort_values('_sort_key').drop('_sort_key', axis=1)
        return df_copy
    else:
        # No model column or index, return as-is
        return df


def get_latest_csv_files(csv_files, split_count=1, verbose=False, model_name=None):
    """
    Group CSV files by name (excluding timestamp) and return only the latest version.

    CSV files are expected to have timestamps in their names:
    - Format 1 (split_count=1): YYYY-MM-DDTHH-MM-SS+ZZZZ_rest_of_name.csv
    - Format 2 (split_count=2): YYYY-MM-DDTHH-MM-SS_YYYY-MM-DDTHH-MM-SS+ZZZZ_rest_of_name.csv

    Args:
        csv_files: List of Path objects for CSV files
        split_count: Number of times to split on '_' to extract timestamp (1 or 2)
        verbose: Whether to print information about multiple versions
        model_name: Model name for verbose output (required if verbose=True)

    Returns:
        List of Path objects for the latest version of each file
    """
    file_groups = defaultdict(list)

    for csv_file in csv_files:
        filename = csv_file.name
        # Extract timestamp based on split_count
        parts = filename.split('_', split_count)

        if len(parts) > split_count:
            timestamp_str = parts[0]
            rest_of_name = '_'.join(parts[split_count:]) if split_count > 1 else parts[1]

            try:
                # Parse timestamp (handle timezone format)
                timestamp = datetime.fromisoformat(timestamp_str.replace('+', '+'))
                file_groups[rest_of_name].append((timestamp, csv_file))
            except (ValueError, TypeError):
                # If timestamp parsing fails, include the file with minimum timestamp
                file_groups[rest_of_name].append((datetime.min, csv_file))

    # Select only the latest file from each group
    latest_files = []
    for rest_of_name, files in file_groups.items():
        # Sort by timestamp and get the latest
        files.sort(key=lambda x: x[0], reverse=True)
        latest_file = files[0][1]
        latest_files.append(latest_file)

        if verbose and len(files) > 1 and model_name:
            print(f"  {model_name}: Using latest file {latest_file.name} (found {len(files)} versions)")

    return latest_files
