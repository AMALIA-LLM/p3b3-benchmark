from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


def load_annotations_from_folder(annotations_dir):
    """Load all annotation CSV files from the annotations folder."""
    csv_files = sorted(Path(annotations_dir).glob("*.csv"))
    dfs = []
    annotator_names = []

    for file in csv_files:
        df = pd.read_csv(file, sep=';', quotechar='"', engine='python')
        dfs.append(df)
        annotator_names.append(file.stem)

    return dfs, annotator_names


def calculate_agreement(rater_a_scores, rater_b_scores):
    """Calculate weighted kappa, MAE, and Pearson correlation between two raters."""
    # Remove any NaN values
    mask = ~(pd.isna(rater_a_scores) | pd.isna(rater_b_scores))
    rater_a_clean = rater_a_scores[mask]
    rater_b_clean = rater_b_scores[mask]

    if len(rater_a_clean) == 0:
        return None, None, None, 0

    # Calculate metrics
    kappa = cohen_kappa_score(rater_a_clean, rater_b_clean, weights='quadratic')
    mae = mean_absolute_error(rater_a_clean, rater_b_clean)
    pearson_corr, _ = pearsonr(rater_a_clean, rater_b_clean)

    return kappa, mae, pearson_corr, len(rater_a_clean)


def calculate_average_annotator_ratings(annotation_dfs):
    """
    Calculate average ratings across all annotators for each conversation.
    Only includes samples where ALL annotators provided valid ratings (not -1, not NaN).
    If any annotator has -1 or NaN for a sample, that entire sample is excluded.
    """
    # Combine all annotations
    all_annotations = []

    for i, df in enumerate(annotation_dfs):
        df_copy = df.copy()
        df_copy['annotator_id'] = i
        all_annotations.append(df_copy)

    # Concatenate all annotations
    combined = pd.concat(all_annotations, ignore_index=True)

    # Mark invalid ratings
    combined['is_valid'] = (combined['rating'] != -1) & (~pd.isna(combined['rating']))

    # Group by sample and check if ALL annotators have valid ratings
    sample_validity = combined.groupby(
        ['conversation_id', 'model_name', 'turn_nr']
    )['is_valid'].all().reset_index()

    sample_validity.columns = ['conversation_id', 'model_name', 'turn_nr', 'all_valid']

    # Keep only samples where all annotators have valid ratings
    valid_samples = sample_validity[sample_validity['all_valid']]

    # Filter combined data to only include fully valid samples
    combined_valid = combined.merge(
        valid_samples[['conversation_id', 'model_name', 'turn_nr']],
        on=['conversation_id', 'model_name', 'turn_nr'],
        how='inner'
    )

    # Calculate mean rating for each sample
    avg_ratings = combined_valid.groupby(
        ['conversation_id', 'model_name', 'turn_nr']
    )['rating'].agg(['mean', 'count', 'std']).reset_index()

    avg_ratings.columns = ['conversation_id', 'model_name', 'turn_nr',
                           'avg_rating', 'n_annotators', 'std_rating']

    return avg_ratings
