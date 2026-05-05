#!/usr/bin/env python3
"""
Calculate agreement between average annotator ratings and LLM judge scores.
Reuses functions from calculate_agreement.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from .utils import calculate_agreement, load_annotations_from_folder, calculate_average_annotator_ratings


def main(judge_file: str):
    # File paths
    annotations_dir = Path("outputs/annotations")

    # Load annotator ratings
    annotation_dfs, annotator_names = load_annotations_from_folder(annotations_dir)

    if len(annotation_dfs) == 0:
        print("Error: No annotation files found in annotations folder.")
        return

    # Load judge scores
    print(f"Loading judge scores from {judge_file}...")
    judge_df = pd.read_csv(judge_file)

    prompt_eval_name = judge_df["prompt-eval-name"].unique()
    judge_name = judge_df["judge_name"].unique()

    # Check if score column exists
    if 'score' not in judge_df.columns:
        print("Error: 'score' column not found in judge file.")
        print(f"Available columns: {', '.join(judge_df.columns)}")
        return

    # Calculate average annotator ratings
    avg_ratings = calculate_average_annotator_ratings(annotation_dfs)

    # Show statistics about average ratings
    print("=" * 60)
    print("Average Annotator Rating Statistics:")
    print("=" * 60)
    print(f"  Mean rating: {avg_ratings['avg_rating'].mean():.2f}")
    print(f"  Std deviation: {avg_ratings['avg_rating'].std():.2f}")
    print(f"  Mean number of annotators per sample: {avg_ratings['n_annotators'].mean():.1f}")
    print(f"  Mean std within samples: {avg_ratings['std_rating'].mean():.2f}")
    print()

    # Show judge score statistics
    print("=" * 60)
    print("Judge Score Statistics:")
    print("=" * 60)
    valid_judge_scores = judge_df['score'][~pd.isna(judge_df['score'])]
    print(f"  Total samples: {len(judge_df)}")
    print(f"  Valid scores: {len(valid_judge_scores)}")
    print(f"  Mean score: {valid_judge_scores.mean():.2f}")
    print(f"  Std deviation: {valid_judge_scores.std():.2f}")
    print()

    # Merge average ratings with judge scores
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
    print(f"  Pearson Correlation: {pearson_corr:.2f}")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print()

    # Also calculate MAE and Pearson on non-rounded values
    mae_continuous = np.abs(merged_clean['avg_rating'] - merged_clean['score']).mean()
    from scipy.stats import pearsonr
    pearson_continuous, _ = pearsonr(merged_clean['avg_rating'], merged_clean['score'])

    print("Continuous (non-rounded) metrics:")
    print(f"  Mean Absolute Error: {mae_continuous:.2f}")
    print(f"  Pearson Correlation: {pearson_continuous:.2f}")
    print()

    output_metrics = {
        'judge_file': judge_file,
        "judge_name": judge_name.tolist(),
        "prompt_eval": prompt_eval_name.tolist(),
        'n_samples': n_samples,
        'kappa': round(kappa, 2),
        'mae': round(mae, 2),
        'pearson_corr': round(pearson_corr, 2),
        'mae_continuous': round(mae_continuous, 2),
        'pearson_continuous': round(pearson_continuous, 2)
    }

    return output_metrics


if __name__ == "__main__":

    # Automatically find all judge files in outputs/annotations_judge
    all_judge_metrics = []

    judge_dir = Path("outputs/annotations_judge")
    judge_files = list(judge_dir.glob("*judge_scored.csv"))

    print(f"Found {len(judge_files)} judge files")

    for j in judge_files:
        print("=" * 60)
        print("Judge File:", j)
        metrics_judge = main(str(j))
        all_judge_metrics.append(metrics_judge)

    # create a df from all_judge
    df = pd.DataFrame.from_dict(all_judge_metrics)
    print("=" * 60)
    print("Summary of all judge metrics:")
    print("=" * 60)
    print(df)
    print()
    df.to_csv("outputs/annotations_judge/judge_metrics_summary.csv", index=False)
