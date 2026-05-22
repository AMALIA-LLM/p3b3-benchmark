import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

from .utils import sort_dataframe_by_model_order, get_latest_csv_files


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _read_csv_files(model_folder: Path, subfolder: str, split_count: int, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Read and concatenate all CSV files from a model's subfolder.

    Args:
        model_folder: Path to the model folder
        subfolder: Name of the subfolder (e.g., 'class_scores', 'llm_scores')
        split_count: Number of underscores expected in filename splits
        verbose: Whether to print verbose output

    Returns:
        Concatenated DataFrame or None if no valid files found
    """
    scores_path = model_folder / subfolder

    if not scores_path.exists():
        return None

    # Find all CSV files and keep only the latest versions
    all_csv_files = list(scores_path.glob("*.csv"))
    csv_files = get_latest_csv_files(
        all_csv_files,
        split_count=split_count,
        verbose=verbose,
        model_name=model_folder.name
    )

    if not csv_files:
        return None

    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def _get_group_combinations(df: pd.DataFrame, has_dataset: bool) -> List[Tuple]:
    """
    Get unique combinations of prompt_type (and dataset if available).

    Args:
        df: DataFrame containing prompt_type and possibly dataset columns
        has_dataset: Whether the dataset column exists

    Returns:
        List of tuples representing group combinations
    """
    if has_dataset:
        return df[['prompt_type', 'dataset']].drop_duplicates().values
    else:
        return [(pt, None) for pt in df['prompt_type'].unique()]


def _filter_by_group(df: pd.DataFrame, group: Tuple, has_dataset: bool) -> pd.DataFrame:
    """
    Filter DataFrame by group (prompt_type and optionally dataset).

    Args:
        df: DataFrame to filter
        group: Tuple of (prompt_type, dataset) or (prompt_type, None)
        has_dataset: Whether the dataset column exists

    Returns:
        Filtered DataFrame
    """
    if has_dataset:
        prompt_type, dataset = group
        return df[(df['prompt_type'] == prompt_type) & (df['dataset'] == dataset)]
    else:
        prompt_type = group[0]
        return df[df['prompt_type'] == prompt_type]


def _calculate_score_averages(df: pd.DataFrame, score_columns: List[str]) -> Dict[str, float]:
    """
    Calculate average scores for each score column.

    Args:
        df: DataFrame containing score columns
        score_columns: List of column names ending with "_score"

    Returns:
        Dictionary mapping classifier names to average scores (rescaled to 0-100)
    """
    scores = {}
    for score_col in score_columns:
        classifier_name = score_col.replace("_score", "")
        numeric_scores = pd.to_numeric(df[score_col], errors='coerce')
        avg_score = numeric_scores.mean()
        scores[classifier_name] = round(avg_score * 100, 1)
    return scores


def _calculate_llm_score_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate LLM score metrics including percentages and filtered scores.

    Args:
        df: DataFrame containing 'score', 'not_pt', and 'invalid' columns

    Returns:
        Dictionary with all LLM score metrics
    """
    metrics = {}

    # Base score
    scores = pd.to_numeric(df['score'], errors='coerce')
    metrics['llm_judge_score'] = round(scores.mean() * 10, 1)
    metrics['num_conversations'] = len(df['conversation_id'].unique()) if 'conversation_id' in df.columns else len(df)

    # not_pt metrics
    if 'not_pt' in df.columns:
        count_not_pt = (df['not_pt'] == True).sum()
        metrics['pct_not_pt'] = round((count_not_pt / len(df)) * 100, 1) if len(df) > 0 else 0.0
        df_no_not_pt = df[df['not_pt'] != True]
        scores_no_not_pt = pd.to_numeric(df_no_not_pt['score'], errors='coerce')
        metrics['llm_judge_score_no_not_pt'] = round(scores_no_not_pt.mean() * 10, 1)
        metrics['num_conversations_no_not_pt'] = len(df_no_not_pt['conversation_id'].unique()) if 'conversation_id' in df_no_not_pt.columns else len(df_no_not_pt)
    else:
        metrics['pct_not_pt'] = 0.0
        metrics['llm_judge_score_no_not_pt'] = metrics['llm_judge_score']
        metrics['num_conversations_no_not_pt'] = metrics['num_conversations']

    # invalid metrics
    if 'invalid' in df.columns:
        count_invalid = (df['invalid'] == True).sum()
        metrics['pct_invalid'] = round((count_invalid / len(df)) * 100, 1) if len(df) > 0 else 0.0
        df_no_invalid = df[df['invalid'] != True]
        scores_no_invalid = pd.to_numeric(df_no_invalid['score'], errors='coerce')
        metrics['llm_judge_score_no_invalid'] = round(scores_no_invalid.mean() * 10, 1)
        metrics['num_conversations_no_invalid'] = len(df_no_invalid['conversation_id'].unique()) if 'conversation_id' in df_no_invalid.columns else len(df_no_invalid)
    else:
        metrics['pct_invalid'] = 0.0
        metrics['llm_judge_score_no_invalid'] = metrics['llm_judge_score']
        metrics['num_conversations_no_invalid'] = metrics['num_conversations']

    # Both excluded
    if 'invalid' in df.columns and 'not_pt' in df.columns:
        df_no_both = df[(df['invalid'] != True) & (df['not_pt'] != True)]
        scores_no_both = pd.to_numeric(df_no_both['score'], errors='coerce')
        metrics['llm_judge_score_no_both'] = round(scores_no_both.mean() * 10, 1)
        metrics['num_conversations_no_both'] = len(df_no_both['conversation_id'].unique()) if 'conversation_id' in df_no_both.columns else len(df_no_both)
    else:
        metrics['llm_judge_score_no_both'] = metrics['llm_judge_score']
        metrics['num_conversations_no_both'] = metrics['num_conversations']

    return metrics


def _organize_results_by_group(df: pd.DataFrame, group_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Organize results by group and return dictionary of DataFrames.

    Args:
        df: DataFrame with results
        group_cols: List of column names to group by (e.g., ['prompt_type', 'dataset'])

    Returns:
        Dictionary mapping group keys to DataFrames
    """
    results = {}
    has_dataset = 'dataset' in df.columns and 'dataset' in group_cols

    if has_dataset:
        # Fill NaN datasets for backwards compatibility
        df['dataset'] = df['dataset'].fillna('all_prompts')
        group_combinations = df[['prompt_type', 'dataset']].drop_duplicates().values

        for prompt_type, dataset in sorted(map(tuple, group_combinations)):
            key = f"{prompt_type}_{dataset}"
            group_df = df[(df['prompt_type'] == prompt_type) & (df['dataset'] == dataset)].copy()
            group_df = group_df.drop(['prompt_type', 'dataset'], axis=1)
            if 'model' in group_df.columns:
                group_df = group_df.set_index('model')
            group_df = sort_dataframe_by_model_order(group_df)
            results[key] = group_df
    else:
        # Group by prompt_type only
        for prompt_type in sorted(df['prompt_type'].unique()):
            group_df = df[df['prompt_type'] == prompt_type].copy()
            group_df = group_df.drop('prompt_type', axis=1)
            if 'model' in group_df.columns:
                group_df = group_df.set_index('model')
            group_df = sort_dataframe_by_model_order(group_df)
            results[prompt_type] = group_df

    return results


# ============================================================================
# MAIN AGGREGATION FUNCTIONS
# ============================================================================


def aggregate_model_scores(results_dir="results"):
    """
    Aggregate scores from all models by averaging classifier scores.

    Returns a DataFrame with models as rows and average scores per classifier.
    """
    results_path = Path(results_dir)
    model_data = []
    first_responses_data = []

    # Iterate through each model folder
    for model_folder in results_path.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        combined_df = _read_csv_files(model_folder, "class_scores", split_count=1, verbose=True)

        if combined_df is None:
            continue

        # Calculate word count for responses
        if 'assistant' in combined_df.columns:
            combined_df['word_count'] = combined_df['assistant'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )

        # Store first responses for each prompt type (and dataset if available)
        if 'prompt_id' in combined_df.columns and 'prompt_type' in combined_df.columns:
            has_dataset = 'dataset' in combined_df.columns
            group_combinations = _get_group_combinations(combined_df, has_dataset)

            for group in group_combinations:
                prompt_df = _filter_by_group(combined_df, group, has_dataset)
                prompt_type = group[0]

                # Get the first prompt (sorted by prompt_id)
                first_prompt = prompt_df.sort_values('prompt_id').iloc[0]

                # Store the response (check for 'assistant' or 'raw_output' column)
                response_text = None
                if 'assistant' in first_prompt and pd.notna(first_prompt['assistant']):
                    response_text = first_prompt['assistant']
                elif 'raw_output' in first_prompt and pd.notna(first_prompt['raw_output']):
                    response_text = first_prompt['raw_output']

                if response_text:
                    response_data = {
                        "model": model_name,
                        "prompt_type": prompt_type,
                        "first_response": response_text,
                        "prompt_id": first_prompt['prompt_id']
                    }
                    if has_dataset:
                        response_data["dataset"] = group[1]
                    first_responses_data.append(response_data)

        # Find all score columns (columns ending with "_score")
        score_columns = [col for col in combined_df.columns if col.endswith("_score")]

        # Group by prompt_type and dataset
        if 'prompt_type' in combined_df.columns:
            has_dataset = 'dataset' in combined_df.columns
            group_combinations = _get_group_combinations(combined_df, has_dataset)

            for group in group_combinations:
                prompt_df = _filter_by_group(combined_df, group, has_dataset)
                prompt_type = group[0]

                model_scores = {"model": model_name, "prompt_type": prompt_type}
                if has_dataset:
                    model_scores["dataset"] = group[1]

                # Calculate average for each score column
                model_scores.update(_calculate_score_averages(prompt_df, score_columns))

                # Calculate average and stdev word count if available
                if 'word_count' in prompt_df.columns:
                    model_scores['avg_word_count'] = prompt_df['word_count'].mean()
                    model_scores['stdev_word_count'] = prompt_df['word_count'].std()

                model_data.append(model_scores)
        else:
            # No prompt_type column, treat as single entry
            has_dataset = 'dataset' in combined_df.columns
            model_scores = {"model": model_name, "prompt_type": "unknown"}
            if has_dataset and len(combined_df) > 0:
                model_scores["dataset"] = combined_df['dataset'].iloc[0]

            model_scores.update(_calculate_score_averages(combined_df, score_columns))

            # Calculate average and stdev word count if available
            if 'word_count' in combined_df.columns:
                model_scores['avg_word_count'] = combined_df['word_count'].mean()
                model_scores['stdev_word_count'] = combined_df['word_count'].std()
            model_data.append(model_scores)

    # Create DataFrame with all data
    all_results_df = pd.DataFrame(model_data)

    # Group by prompt_type (and dataset if available) and create separate tables
    results_by_prompt = _organize_results_by_group(
        all_results_df,
        group_cols=['prompt_type', 'dataset'] if 'dataset' in all_results_df.columns else ['prompt_type']
    )

    # Process first responses
    first_responses_dict = {}
    if first_responses_data:
        first_resp_df = pd.DataFrame(first_responses_data)
        has_dataset = 'dataset' in first_resp_df.columns

        group_combinations = _get_group_combinations(first_resp_df, has_dataset)
        for group in group_combinations:
            prompt_type = group[0]
            prompt_responses = _filter_by_group(first_resp_df, group, has_dataset)
            key = f"{prompt_type}_{group[1]}" if has_dataset else prompt_type
            first_responses_dict[key] = prompt_responses

    return results_by_prompt, first_responses_dict

def aggregate_scores_by_turn(results_dir="results"):
    """
    Aggregate scores by turn for each model, classifier and dataset.

    Returns a dictionary with per-turn DataFrames organized by group.
    """
    results_path = Path(results_dir)
    turn_data = []

    # Iterate through each model folder
    for model_folder in results_path.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        combined_df = _read_csv_files(model_folder, "class_scores", split_count=1, verbose=False)

        if combined_df is None:
            continue

        # Check if turn information is available
        if 'turn_nr' not in combined_df.columns:
            print(f"Warning: No 'turn_nr' column found in {model_name} data. Skipping turn analysis.")
            continue

        # Find all score columns (columns ending with "_score")
        score_columns = [col for col in combined_df.columns if col.endswith("_score")]

        # Group by prompt_type, dataset, and turn
        if 'prompt_type' in combined_df.columns:
            has_dataset = 'dataset' in combined_df.columns
            group_combinations = _get_group_combinations(combined_df, has_dataset)

            for group in group_combinations:
                prompt_df = _filter_by_group(combined_df, group, has_dataset)
                prompt_type = group[0]

                # Group by turn
                for turn in sorted(prompt_df['turn_nr'].unique()):
                    turn_df = prompt_df[prompt_df['turn_nr'] == turn]

                    turn_scores = {
                        "model": model_name,
                        "prompt_type": prompt_type,
                        "turn_nr": turn
                    }
                    if has_dataset:
                        turn_scores["dataset"] = group[1]

                    # Calculate average for each score column
                    turn_scores.update(_calculate_score_averages(turn_df, score_columns))

                    turn_data.append(turn_scores)

    # Create DataFrame with all data
    turn_results_df = pd.DataFrame(turn_data)

    if turn_results_df.empty:
        return {}

    # Group by prompt_type, dataset (if available), and turn
    results_by_group_turn = {}

    has_dataset = 'dataset' in turn_results_df.columns

    if has_dataset:
        # if dataset is NaN replace by all_prompts because of backwards compatibility
        turn_results_df['dataset'] = turn_results_df['dataset'].fillna('all_prompts')

        # Group by prompt_type, dataset, and turn
        group_combinations = turn_results_df[['prompt_type', 'dataset']].drop_duplicates().values
        for prompt_type, dataset in sorted(map(tuple, group_combinations)):
            key = f"{prompt_type}_{dataset}"
            group_df = turn_results_df[(turn_results_df['prompt_type'] == prompt_type) &
                                      (turn_results_df['dataset'] == dataset)].copy()

            # Create pivot table with turns as columns
            score_cols = [col for col in group_df.columns
                         if col not in ['model', 'prompt_type', 'dataset', 'turn_nr']]

            # For each classifier, create a separate view
            for classifier in score_cols:
                pivot_df = group_df.pivot_table(
                    index='model',
                    columns='turn_nr',
                    values=classifier,
                    aggfunc='mean'
                )
                pivot_df.columns = [f"turn_{int(col)}" for col in pivot_df.columns]
                pivot_df = sort_dataframe_by_model_order(pivot_df)

                classifier_key = f"{key}_{classifier}"
                results_by_group_turn[classifier_key] = pivot_df
    else:
        # Group by prompt_type and turn only
        for prompt_type in sorted(turn_results_df['prompt_type'].unique()):
            group_df = turn_results_df[turn_results_df['prompt_type'] == prompt_type].copy()

            # Create pivot table with turns as columns
            score_cols = [col for col in group_df.columns
                         if col not in ['model', 'prompt_type', 'turn_nr']]

            # For each classifier, create a separate view
            for classifier in score_cols:
                pivot_df = group_df.pivot_table(
                    index='model',
                    columns='turn_nr',
                    values=classifier,
                    aggfunc='mean'
                )
                pivot_df.columns = [f"turn_{int(col)}" for col in pivot_df.columns]
                pivot_df = sort_dataframe_by_model_order(pivot_df)

                classifier_key = f"{prompt_type}_{classifier}"
                results_by_group_turn[classifier_key] = pivot_df

    return results_by_group_turn

def aggregate_llm_scores(results_dir="results", llm_scores_subfolder="llm_scores"):
    """
    Aggregate LLM scores from all models.

    Args:
        results_dir: Path to the results directory
        llm_scores_subfolder: Name of the subfolder containing LLM scores (default: "llm_scores")

    Returns a dictionary with DataFrames organized by prompt_type (and dataset if available).
    """
    results_path = Path(results_dir)
    llm_data = []

    # Iterate through each model folder
    for model_folder in results_path.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        combined_df = _read_csv_files(model_folder, llm_scores_subfolder, split_count=2, verbose=True)

        if combined_df is None:
            continue

        # Check for required columns
        if 'score' not in combined_df.columns or 'prompt_type' not in combined_df.columns:
            print(f"Warning: Missing required columns in {model_name} LLM scores. Skipping.")
            continue

        # Group by prompt_type
        for prompt_type in combined_df['prompt_type'].unique():
            prompt_df = combined_df[combined_df['prompt_type'] == prompt_type]

            # Calculate all LLM metrics using helper
            metrics = _calculate_llm_score_metrics(prompt_df)

            llm_scores = {
                "model": model_name,
                "prompt_type": prompt_type,
                **metrics
            }

            llm_data.append(llm_scores)

    # Create DataFrame with all data
    llm_results_df = pd.DataFrame(llm_data)

    if llm_results_df.empty:
        return {}, {}, {}, {}

    # Group by prompt_type and create separate tables
    results_by_prompt = {}
    results_by_prompt_no_not_pt = {}
    results_by_prompt_no_invalid = {}
    results_by_prompt_no_both = {}

    for prompt_type in sorted(llm_results_df['prompt_type'].unique()):
        prompt_df = llm_results_df[llm_results_df['prompt_type'] == prompt_type].copy()

        # Create all scores table
        prompt_df_all = prompt_df[['model', 'llm_judge_score', 'num_conversations', 'pct_not_pt', 'pct_invalid']].copy()
        prompt_df_all = prompt_df_all.set_index('model')
        prompt_df_all = sort_dataframe_by_model_order(prompt_df_all)
        results_by_prompt[prompt_type] = prompt_df_all

        # Create no_not_pt table
        prompt_df_no_not_pt = prompt_df[['model', 'llm_judge_score_no_not_pt', 'num_conversations_no_not_pt', 'pct_not_pt']].copy()
        prompt_df_no_not_pt.columns = ['model', 'llm_judge_score', 'num_conversations', 'pct_not_pt']
        prompt_df_no_not_pt = prompt_df_no_not_pt.set_index('model')
        prompt_df_no_not_pt = sort_dataframe_by_model_order(prompt_df_no_not_pt)
        results_by_prompt_no_not_pt[prompt_type] = prompt_df_no_not_pt

        # Create no_invalid table
        prompt_df_no_invalid = prompt_df[['model', 'llm_judge_score_no_invalid', 'num_conversations_no_invalid', 'pct_invalid']].copy()
        prompt_df_no_invalid.columns = ['model', 'llm_judge_score', 'num_conversations', 'pct_invalid']
        prompt_df_no_invalid = prompt_df_no_invalid.set_index('model')
        prompt_df_no_invalid = sort_dataframe_by_model_order(prompt_df_no_invalid)
        results_by_prompt_no_invalid[prompt_type] = prompt_df_no_invalid

        # Create no_both table (excluding both invalid=True AND not_pt=True)
        prompt_df_no_both = prompt_df[['model', 'llm_judge_score_no_both', 'num_conversations_no_both']].copy()
        prompt_df_no_both.columns = ['model', 'llm_judge_score', 'num_conversations']
        prompt_df_no_both = prompt_df_no_both.set_index('model')
        prompt_df_no_both = sort_dataframe_by_model_order(prompt_df_no_both)
        results_by_prompt_no_both[prompt_type] = prompt_df_no_both

    return results_by_prompt, results_by_prompt_no_not_pt, results_by_prompt_no_invalid, results_by_prompt_no_both

def aggregate_combined_table(results_dir="results", llm_scores_subfolder="llm_scores"):
    """
    Create a single comprehensive table combining classifier scores, LLM scores, and percentages.

    Args:
        results_dir: Path to the results directory
        llm_scores_subfolder: Name of the subfolder containing LLM scores (default: "llm_scores")

    Returns a DataFrame with:
    - Rows: models
    - Columns organized by prompt_type (normal, pt-pt, pt-br):
        - classifier scores (for each classifier)
        - llm_judge_score (excluding invalid and non_pt)
        - pct_not_pt
        - pct_invalid
    """
    results_path = Path(results_dir)

    # First, get classifier scores
    classifier_data = {}
    for model_folder in results_path.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        combined_df = _read_csv_files(model_folder, "class_scores", split_count=1, verbose=False)

        if combined_df is None:
            continue

        # Find all score columns
        score_columns = [col for col in combined_df.columns if col.endswith("_score")]

        # Group by prompt_type
        if 'prompt_type' in combined_df.columns:
            for prompt_type in combined_df['prompt_type'].unique():
                prompt_df = combined_df[combined_df['prompt_type'] == prompt_type]

                if model_name not in classifier_data:
                    classifier_data[model_name] = {}
                if prompt_type not in classifier_data[model_name]:
                    classifier_data[model_name][prompt_type] = {}

                # Calculate average for each score column
                classifier_data[model_name][prompt_type] = _calculate_score_averages(prompt_df, score_columns)

    # Now get LLM scores
    llm_data = {}
    for model_folder in results_path.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        combined_df = _read_csv_files(model_folder, llm_scores_subfolder, split_count=2, verbose=False)

        if combined_df is None:
            continue

        # Check for required columns
        if 'score' not in combined_df.columns or 'prompt_type' not in combined_df.columns:
            continue

        # Group by prompt_type
        for prompt_type in combined_df['prompt_type'].unique():
            prompt_df = combined_df[combined_df['prompt_type'] == prompt_type]

            if model_name not in llm_data:
                llm_data[model_name] = {}
            if prompt_type not in llm_data[model_name]:
                llm_data[model_name][prompt_type] = {}

            # Calculate percentages
            if 'not_pt' in prompt_df.columns:
                count_not_pt = (prompt_df['not_pt'] == True).sum()
                pct_not_pt = round((count_not_pt / len(prompt_df)) * 100, 1) if len(prompt_df) > 0 else 0.0
            else:
                pct_not_pt = 0.0

            if 'invalid' in prompt_df.columns:
                count_invalid = (prompt_df['invalid'] == True).sum()
                pct_invalid = round((count_invalid / len(prompt_df)) * 100, 1) if len(prompt_df) > 0 else 0.0
            else:
                pct_invalid = 0.0

            # Calculate score excluding only not_pt=True
            if 'not_pt' in prompt_df.columns:
                prompt_df_no_not_pt = prompt_df[prompt_df['not_pt'] != True]
                scores_no_not_pt = pd.to_numeric(prompt_df_no_not_pt['score'], errors='coerce')
                avg_score_no_not_pt = round(scores_no_not_pt.mean() * 10, 1)
            else:
                scores = pd.to_numeric(prompt_df['score'], errors='coerce')
                avg_score_no_not_pt = round(scores.mean() * 10, 1)

            llm_data[model_name][prompt_type]['llm_judge_score'] = avg_score_no_not_pt
            llm_data[model_name][prompt_type]['pct_not_pt'] = pct_not_pt
            llm_data[model_name][prompt_type]['pct_invalid'] = pct_invalid

    # Combine into single DataFrame
    rows = []
    all_models = set(classifier_data.keys()) | set(llm_data.keys())

    for model in all_models:
        row = {'model': model}

        # Get all prompt types for this model
        prompt_types = set()
        if model in classifier_data:
            prompt_types.update(classifier_data[model].keys())
        if model in llm_data:
            prompt_types.update(llm_data[model].keys())

        # For each prompt type, add columns
        for prompt_type in prompt_types:
            prefix = f"{prompt_type}_"

            # Add classifier scores
            if model in classifier_data and prompt_type in classifier_data[model]:
                for classifier_name, score in classifier_data[model][prompt_type].items():
                    row[f"{prefix}{classifier_name}"] = score

            # Add LLM judge score and percentages
            if model in llm_data and prompt_type in llm_data[model]:
                row[f"{prefix}llm_judge_score"] = llm_data[model][prompt_type]['llm_judge_score']
                row[f"{prefix}pct_not_pt"] = llm_data[model][prompt_type]['pct_not_pt']
                row[f"{prefix}pct_invalid"] = llm_data[model][prompt_type]['pct_invalid']

        rows.append(row)

    # Create DataFrame
    combined_df = pd.DataFrame(rows)
    combined_df = combined_df.set_index('model')
    combined_df = sort_dataframe_by_model_order(combined_df)

    # Sort columns: for each prompt_type, group related columns together
    if len(combined_df.columns) > 0:
        # Extract prompt types from column names
        prompt_types_found = set()
        for col in combined_df.columns:
            for pt in ['normal', 'pt-pt', 'pt-br']:
                if col.startswith(f"{pt}_"):
                    prompt_types_found.add(pt)
                    break

        # Sort columns by prompt type
        sorted_cols = []
        for pt in sorted(prompt_types_found):
            # Get all columns for this prompt type
            pt_cols = [col for col in combined_df.columns if col.startswith(f"{pt}_")]
            # Sort them so classifiers come first, then llm, then percentages
            classifier_cols = [col for col in pt_cols if not any(x in col for x in ['llm_judge_score', 'pct_not_pt', 'pct_invalid'])]
            llm_col = [col for col in pt_cols if 'llm_judge_score' in col]
            pct_cols = [col for col in pt_cols if 'pct_' in col]
            sorted_cols.extend(sorted(classifier_cols))
            sorted_cols.extend(llm_col)
            sorted_cols.extend(sorted(pct_cols))

        combined_df = combined_df[sorted_cols]

    return combined_df

def aggregate_llm_scores_by_turn(results_dir="results", llm_scores_subfolder="llm_scores"):
    """
    Aggregate LLM scores by turn for each model.

    Args:
        results_dir: Path to the results directory
        llm_scores_subfolder: Name of the subfolder containing LLM scores (default: "llm_scores")

    Returns a dictionary with per-turn DataFrames organized by prompt_type.
    """
    results_path = Path(results_dir)
    turn_data = []

    # Iterate through each model folder
    for model_folder in results_path.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        combined_df = _read_csv_files(model_folder, llm_scores_subfolder, split_count=2, verbose=False)

        if combined_df is None:
            continue

        # Check for required columns
        if 'score' not in combined_df.columns or 'prompt_type' not in combined_df.columns or 'turn_nr' not in combined_df.columns:
            print(f"Warning: Missing required columns in {model_name} LLM scores for turn analysis. Skipping.")
            continue

        # Group by prompt_type and turn
        for prompt_type in combined_df['prompt_type'].unique():
            prompt_df = combined_df[combined_df['prompt_type'] == prompt_type]

            # Group by turn
            for turn in sorted(prompt_df['turn_nr'].unique()):
                turn_df = prompt_df[prompt_df['turn_nr'] == turn]

                # Calculate all metrics using helper (temporarily without conversation_id to get subset metrics)
                metrics = _calculate_llm_score_metrics(turn_df)

                turn_scores = {
                    "model": model_name,
                    "prompt_type": prompt_type,
                    "turn_nr": turn,
                    "llm_judge_score": metrics['llm_judge_score'],
                    "pct_not_pt": metrics['pct_not_pt'],
                    "pct_invalid": metrics['pct_invalid'],
                    "llm_judge_score_no_not_pt": metrics['llm_judge_score_no_not_pt'],
                    "llm_judge_score_no_invalid": metrics['llm_judge_score_no_invalid'],
                    "llm_judge_score_no_both": metrics['llm_judge_score_no_both']
                }

                turn_data.append(turn_scores)

    # Create DataFrame with all data
    turn_results_df = pd.DataFrame(turn_data)

    if turn_results_df.empty:
        return {}, {}, {}, {}

    # Group by prompt_type and create pivot tables
    results_by_prompt = {}
    results_by_prompt_no_not_pt = {}
    results_by_prompt_no_invalid = {}
    results_by_prompt_no_both = {}

    for prompt_type in sorted(turn_results_df['prompt_type'].unique()):
        prompt_df = turn_results_df[turn_results_df['prompt_type'] == prompt_type].copy()

        # Create pivot table with turns as columns (all data)
        pivot_df = prompt_df.pivot_table(
            index='model',
            columns='turn_nr',
            values='llm_judge_score',
            aggfunc='mean'
        )
        pivot_df.columns = [f"turn_{int(col)}" for col in pivot_df.columns]
        pivot_df = sort_dataframe_by_model_order(pivot_df)
        results_by_prompt[prompt_type] = pivot_df

        # Create pivot table excluding not_pt=True
        pivot_df_no_not_pt = prompt_df.pivot_table(
            index='model',
            columns='turn_nr',
            values='llm_judge_score_no_not_pt',
            aggfunc='mean'
        )
        pivot_df_no_not_pt.columns = [f"turn_{int(col)}" for col in pivot_df_no_not_pt.columns]
        pivot_df_no_not_pt = sort_dataframe_by_model_order(pivot_df_no_not_pt)
        results_by_prompt_no_not_pt[prompt_type] = pivot_df_no_not_pt

        # Create pivot table excluding invalid=True
        pivot_df_no_invalid = prompt_df.pivot_table(
            index='model',
            columns='turn_nr',
            values='llm_judge_score_no_invalid',
            aggfunc='mean'
        )
        pivot_df_no_invalid.columns = [f"turn_{int(col)}" for col in pivot_df_no_invalid.columns]
        pivot_df_no_invalid = sort_dataframe_by_model_order(pivot_df_no_invalid)
        results_by_prompt_no_invalid[prompt_type] = pivot_df_no_invalid

        # Create pivot table excluding both invalid=True AND not_pt=True
        pivot_df_no_both = prompt_df.pivot_table(
            index='model',
            columns='turn_nr',
            values='llm_judge_score_no_both',
            aggfunc='mean'
        )
        pivot_df_no_both.columns = [f"turn_{int(col)}" for col in pivot_df_no_both.columns]
        pivot_df_no_both = sort_dataframe_by_model_order(pivot_df_no_both)
        results_by_prompt_no_both[prompt_type] = pivot_df_no_both

    return results_by_prompt, results_by_prompt_no_not_pt, results_by_prompt_no_invalid, results_by_prompt_no_both


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Aggregate model evaluation results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to the results directory (default: results)"
    )
    parser.add_argument(
        "--llm_scores_subfolder",
        type=str,
        default="llm_scores",
        help="Name of the subfolder containing LLM scores (default: llm_scores)"
    )
    args = parser.parse_args()

    # Define results directory
    results_dir = args.results_dir
    llm_scores_subfolder = args.llm_scores_subfolder
    results_path = Path(results_dir)

    # Create output directory structure
    classifier_general_path = results_path / "z_classifier_scores" / "general"
    classifier_turn_path = results_path / "z_classifier_scores" / "turn_level"
    llm_general_path = results_path / f"z_{llm_scores_subfolder}" / "general"
    llm_turn_path = results_path / f"z_{llm_scores_subfolder}" / "turn_level"

    classifier_general_path.mkdir(parents=True, exist_ok=True)
    classifier_turn_path.mkdir(parents=True, exist_ok=True)
    llm_general_path.mkdir(parents=True, exist_ok=True)
    llm_turn_path.mkdir(parents=True, exist_ok=True)

    # Create combined comprehensive table
    print("\n" + "="*80)
    print("GENERATING COMBINED COMPREHENSIVE TABLE")
    print("="*80 + "\n")

    combined_table = aggregate_combined_table(results_dir, llm_scores_subfolder)

    if not combined_table.empty:
        print("COMBINED SCORES - ALL PROMPT TYPES")
        print("="*80 + "\n")
        print(combined_table.to_string())
        print("\n")

        # Save to CSV
        output_file = results_path / f"combined_comprehensive_scores_{llm_scores_subfolder}.csv"
        combined_table.to_csv(output_file)
        print(f"Combined table saved to: {output_file}\n")
    else:
        print("No data available for combined table.\n")

    # Aggregate results
    results_by_prompt, first_responses_dict = aggregate_model_scores(results_dir)

    # Print with formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda x: f'{x:.1f}')

    # Display first responses
    if first_responses_dict:
        print("\n" + "="*80)
        print("FIRST RESPONSES TO FIRST PROMPT")
        print("="*80 + "\n")

        for key in sorted(first_responses_dict.keys()):
            responses = first_responses_dict[key]
            print(f"\n--- GROUP: {key.upper()} ---\n")

            for _, row in responses.iterrows():
                print(f"Model: {row['model']}")
                print(f"Prompt ID: {row['prompt_id']}")
                print(f"Response: {row['first_response'][:500]}...")  # First 500 chars
                print("-" * 80)
                print()

    # Display results for each group
    for key in sorted(results_by_prompt.keys()):
        results = results_by_prompt[key]

        print("\n" + "="*80)
        print(f"SCORES - GROUP: {key.upper()}")
        print("="*80 + "\n")
        print(results.to_string())
        print("\n")

        # Save to CSV in classifier_scores/general directory
        output_file = classifier_general_path / f"aggregated_scores_{key}.csv"
        results.to_csv(output_file)
        print(f"Results saved to: {output_file}\n")

    # Aggregate and display per-turn results
    print("\n" + "="*80)
    print("GENERATING PER-TURN RESULTS")
    print("="*80 + "\n")

    turn_results = aggregate_scores_by_turn(results_dir)

    if turn_results:
        for key in sorted(turn_results.keys()):
            results = turn_results[key]

            print("\n" + "="*80)
            print(f"PER-TURN SCORES - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in classifier_scores/turn_level directory
            output_file = classifier_turn_path / f"aggregated_scores_by_turn_{key}.csv"
            results.to_csv(output_file)
            print(f"Per-turn results saved to: {output_file}\n")
    else:
        print("No per-turn data available (turn column not found in results).\n")

    # Aggregate and display LLM scores
    print("\n" + "="*80)
    print("GENERATING LLM JUDGE RESULTS")
    print("="*80 + "\n")

    llm_results, llm_results_no_not_pt, llm_results_no_invalid, llm_results_no_both = aggregate_llm_scores(results_dir, llm_scores_subfolder)

    if llm_results:
        # Display all scores
        for key in sorted(llm_results.keys()):
            results = llm_results[key]

            print("\n" + "="*80)
            print(f"LLM JUDGE SCORES (ALL) - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in llm_scores/general directory
            output_file = llm_general_path / f"aggregated_llm_scores_{key}.csv"
            results.to_csv(output_file)
            print(f"LLM judge results saved to: {output_file}\n")

        # Display scores excluding not_pt=True
        for key in sorted(llm_results_no_not_pt.keys()):
            results = llm_results_no_not_pt[key]

            print("\n" + "="*80)
            print(f"LLM JUDGE SCORES (EXCLUDING not_pt=True) - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in llm_scores/general directory
            output_file = llm_general_path / f"aggregated_llm_scores_{key}_no_not_pt.csv"
            results.to_csv(output_file)
            print(f"LLM judge results (no not_pt) saved to: {output_file}\n")

        # Display scores excluding invalid=True
        for key in sorted(llm_results_no_invalid.keys()):
            results = llm_results_no_invalid[key]

            print("\n" + "="*80)
            print(f"LLM JUDGE SCORES (EXCLUDING invalid=True) - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in llm_scores/general directory
            output_file = llm_general_path / f"aggregated_llm_scores_{key}_no_invalid.csv"
            results.to_csv(output_file)
            print(f"LLM judge results (no invalid) saved to: {output_file}\n")

        # Display scores excluding both invalid=True AND not_pt=True
        for key in sorted(llm_results_no_both.keys()):
            results = llm_results_no_both[key]

            print("\n" + "="*80)
            print(f"LLM JUDGE SCORES (EXCLUDING invalid=True AND not_pt=True) - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in llm_scores/general directory
            output_file = llm_general_path / f"aggregated_llm_scores_{key}_no_invalid_and_no_not_pt.csv"
            results.to_csv(output_file)
            print(f"LLM judge results (no invalid and no not_pt) saved to: {output_file}\n")
    else:
        print("No LLM judge scores available.\n")

    # Aggregate and display per-turn LLM scores
    print("\n" + "="*80)
    print("GENERATING PER-TURN LLM JUDGE RESULTS")
    print("="*80 + "\n")

    llm_turn_results, llm_turn_results_no_not_pt, llm_turn_results_no_invalid, llm_turn_results_no_both = aggregate_llm_scores_by_turn(results_dir, llm_scores_subfolder)

    if llm_turn_results:
        # Display all scores
        for key in sorted(llm_turn_results.keys()):
            results = llm_turn_results[key]

            print("\n" + "="*80)
            print(f"PER-TURN LLM JUDGE SCORES (ALL) - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in llm_scores/turn_level directory
            output_file = llm_turn_path / f"aggregated_llm_scores_by_turn_{key}.csv"
            results.to_csv(output_file)
            print(f"Per-turn LLM judge results saved to: {output_file}\n")

        # Display scores excluding not_pt=True
        for key in sorted(llm_turn_results_no_not_pt.keys()):
            results = llm_turn_results_no_not_pt[key]

            print("\n" + "="*80)
            print(f"PER-TURN LLM JUDGE SCORES (EXCLUDING not_pt=True) - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in llm_scores/turn_level directory
            output_file = llm_turn_path / f"aggregated_llm_scores_by_turn_{key}_no_not_pt.csv"
            results.to_csv(output_file)
            print(f"Per-turn LLM judge results (no not_pt) saved to: {output_file}\n")

        # Display scores excluding invalid=True
        for key in sorted(llm_turn_results_no_invalid.keys()):
            results = llm_turn_results_no_invalid[key]

            print("\n" + "="*80)
            print(f"PER-TURN LLM JUDGE SCORES (EXCLUDING invalid=True) - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in llm_scores/turn_level directory
            output_file = llm_turn_path / f"aggregated_llm_scores_by_turn_{key}_no_invalid.csv"
            results.to_csv(output_file)
            print(f"Per-turn LLM judge results (no invalid) saved to: {output_file}\n")

        # Display scores excluding both invalid=True AND not_pt=True
        for key in sorted(llm_turn_results_no_both.keys()):
            results = llm_turn_results_no_both[key]

            print("\n" + "="*80)
            print(f"PER-TURN LLM JUDGE SCORES (EXCLUDING invalid=True AND not_pt=True) - {key.upper()}")
            print("="*80 + "\n")
            print(results.to_string())
            print("\n")

            # Save to CSV in llm_scores/turn_level directory
            output_file = llm_turn_path / f"aggregated_llm_scores_by_turn_{key}_no_invalid_and_no_not_pt.csv"
            results.to_csv(output_file)
            print(f"Per-turn LLM judge results (no invalid and no not_pt) saved to: {output_file}\n")
    else:
        print("No per-turn LLM judge scores available.\n")


# example usage:
# python -m src.analysis.aggregation
