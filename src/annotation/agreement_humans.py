#!/usr/bin/env python3
"""
Calculate inter-rater agreement metrics from annotation CSV files.
Computes Weighted Kappa (Quadratic) and Mean Absolute Error (MAE).
"""

import pandas as pd
import numpy as np

from .utils import calculate_agreement, load_annotations_from_folder


def main():
    # Find all CSV files in annotations folder

    dfs, annotator_names = load_annotations_from_folder("outputs/annotations")

    # Show individual annotator statistics
    print("=" * 60)
    print("Individual Annotator Statistics:")
    print("=" * 60)
    for i, (df, annotator_name) in enumerate(zip(dfs, annotator_names)):

        # Count different types of values
        total_samples = len(df)
        num_minus_one = (df['rating'] == -1).sum()
        num_nan = pd.isna(df['rating']).sum()

        # Filter out -1, NaN, and empty values
        valid_ratings = df['rating'][
            (df['rating'] != -1) &
            (~pd.isna(df['rating']))
        ]

        print(f"{annotator_name}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Valid samples: {len(valid_ratings)}")
        print(f"  -1 values: {num_minus_one}")
        print(f"  NaN/empty values: {num_nan}")

        if len(valid_ratings) > 0:
            mean_rating = valid_ratings.mean()
            std_rating = valid_ratings.std()
            print(f"  Mean rating: {mean_rating:.2f}")
            print(f"  Std deviation: {std_rating:.2f}")
        print()

    print("=" * 60)
    print()

    # Calculate pairwise agreement between all raters
    results = []

    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):
            rater_a_name = annotator_names[i]
            rater_b_name = annotator_names[j]

            # Merge on conversation_id to align ratings
            merged = dfs[i].merge(
                dfs[j],
                on=['conversation_id', 'model_name', 'turn_nr'],
                suffixes=('_a', '_b'),
                how='inner'
            )

            if 'rating_a' not in merged.columns or 'rating_b' not in merged.columns:
                print(f"Warning: Could not find rating columns for {rater_a_name} and {rater_b_name}")
                continue

            # Remove rows where any rating is -1, empty, or NaN
            merged = merged[
                (merged['rating_a'] != -1) &
                (merged['rating_b'] != -1) &
                (~pd.isna(merged['rating_a'])) &
                (~pd.isna(merged['rating_b']))
            ]

            # Calculate agreement
            kappa, mae, pearson_corr, n_samples = calculate_agreement(
                merged['rating_a'],
                merged['rating_b']
            )

            if kappa is not None:
                results.append({
                    'rater_a': rater_a_name,
                    'rater_b': rater_b_name,
                    'n_samples': n_samples,
                    'weighted_kappa': kappa,
                    'mae': mae,
                    'pearson_correlation': pearson_corr
                })

                print(f"Agreement between {rater_a_name} and {rater_b_name}:")
                print(f"  Samples: {n_samples}")
                print(f"  Weighted Kappa (Quadratic): {kappa:.2f}")
                print(f"  Mean Absolute Error: {mae:.2f}")
                print(f"  Pearson Correlation: {pearson_corr:.2f}")
                print()

    # Summary statistics if more than 2 raters
    if len(results) > 1:
        print("=" * 60)
        print("Overall Statistics (across all rater pairs):")
        kappas = [r['weighted_kappa'] for r in results]
        maes = [r['mae'] for r in results]
        pearson_corrs = [r['pearson_correlation'] for r in results]
        print(f"  Mean Weighted Kappa: {np.mean(kappas):.2f} (±{np.std(kappas):.2f})")
        print(f"  Mean MAE: {np.mean(maes):.2f} (±{np.std(maes):.2f})")
        print(f"  Mean Pearson Correlation: {np.mean(pearson_corrs):.2f} (±{np.std(pearson_corrs):.2f})")
        print()


if __name__ == "__main__":
    main()
