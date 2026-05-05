from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .utils import calculate_agreement, load_annotations_from_folder, calculate_average_annotator_ratings


# Measure the agreement between the users and the classifiers
def main():

    folder = Path("results")
    # grab all csv files inside subfolder named class scores
    csv_files = sorted(folder.glob("*/class_scores/*.csv"))

    # read all files to the same dataframe
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)

    # join in the sama dataframe
    combined_df = pd.concat(all_data, ignore_index=True)

    # rename columns to match the ones in the annotation sheet
    combined_df["conversation_id"] = combined_df["prompt_id"]

    # sum one to all turn_nr
    combined_df["turn_nr"] = combined_df["turn_nr"] + 1

    # for the scoring columns we multiply by 10 and round
    combined_df["PtVId_score_scaled"] = round(combined_df["PtVId_score"] * 10)
    combined_df["PeroVaz_PT - BR_Classifier_score_scaled"] = round(combined_df["PeroVaz_PT-BR_Classifier_score"] * 10)

    # save each one to a dataframe wit columns
    # prompt-eval-name, prompt, conversation_id, model_name, prompt_type, turn_nr, accumulate_context, context, response, judge_name, raw_judge_input, llm_evaluation_raw, reasoning, score, not_pt, invalid
    ptvid_df = combined_df[["raw_output", "conversation_id", "model_name", "prompt_type", "turn_nr", "user", "assistant", "PtVId_score_scaled"]]
    # rename columns to match the ones in the annotation sheet
    ptvid_df.rename(columns={
        "raw_output": "prompt",
        "conversation_id": "conversation_id",
        "user": "context",
        "assistant": "response",
        "PtVId_score_scaled": "score",
    }, inplace=True)

    # add a new column named prompt-eval-name
    ptvid_df["prompt_eval_name"] = "ptvid"

    # now the same for perovaz
    pero_vaz_df = combined_df[
        ["raw_output", "conversation_id", "model_name", "prompt_type", "turn_nr", "user", "assistant", "PeroVaz_PT - BR_Classifier_score_scaled"]]
    # rename columns to match the ones in the annotation sheet
    pero_vaz_df.rename(columns={
        "raw_output": "prompt",
        "conversation_id": "conversation_id",
        "user": "context",
        "assistant": "response",
        "PeroVaz_PT - BR_Classifier_score_scaled": "score",
    }, inplace=True)

    # add a new column named prompt-eval-name
    pero_vaz_df["prompt_eval_name"] = "per_vaz_df"

    # calculate average score
    annotations_dir = Path("outputs/annotations")

    annotation_dfs, annotator_names = load_annotations_from_folder(annotations_dir)

    avg_ratings = calculate_average_annotator_ratings(annotation_dfs)

    for judge_df in [ptvid_df, pero_vaz_df]:

        merged = avg_ratings.merge(
            judge_df[['conversation_id', 'model_name', 'turn_nr', 'score']],
            on=['conversation_id', 'model_name', 'turn_nr'],
            how='inner'
        )

        # Remove NaN scores
        merged_clean = merged[~pd.isna(merged['score'])].copy()

        if len(merged_clean) == 0:
            print("Error: No valid samples after merging and cleaning.")
            return

        print(f"Valid samples for agreement calculation: {len(merged_clean)}")
        print()

        # Round average ratings to nearest integer for kappa calculation
        merged_clean['avg_rating_rounded'] = merged_clean['avg_rating'].round().astype(int)
        merged_clean['score_rounded'] = merged_clean['score'].round().astype(int)

        # Calculate agreement using rounded values for kappa
        print("=" * 60)
        print("Agreement Between Average Annotator Ratings and Judge Scores:")
        print("=" * 60)

        kappa, mae, pearson_corr, n_samples = calculate_agreement(
            merged_clean['avg_rating_rounded'],
            merged_clean['score_rounded']
        )

        print(f"  Samples: {n_samples}")
        print(f"  Weighted Kappa (Quadratic): {kappa:.2f}")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  Pearson Correlation: {pearson_corr:.2f}")
        print()

        # Also calculate MAE and Pearson on non-rounded values
        mae_continuous = np.abs(merged_clean['avg_rating'] - merged_clean['score']).mean()

        pearson_continuous, _ = pearsonr(merged_clean['avg_rating'], merged_clean['score'])

        print("Continuous (non-rounded) metrics:")
        print(f"  Mean Absolute Error: {mae_continuous:.2f}")
        print(f"  Pearson Correlation: {pearson_continuous:.2f}")
        print()

        print("="*20)


if __name__ == "__main__":
    main()
