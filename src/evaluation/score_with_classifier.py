import math, json, transformers, pandas as pd, numpy as np, argparse
from os import path, makedirs

from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import cast
from glob import glob
from itertools import batched


class Classifier:

    model_name: str
    pipe: transformers.Pipeline
    max_length: int

    def __init__(self, model_name : str, supports_batching=False):
        self.model_name = model_name
        self.pipe = transformers.pipeline('text-classification', model=model_name, top_k=None)
        self.max_length = self.pipe.model.config.max_position_embeddings or 4096
        self.supports_batching = supports_batching


def classify_responses(cls : Classifier, responses) -> list[dict]:
    """
    Classify a list of text responses using a transformer model, handling text splitting for long inputs.

    This function processes responses that may exceed the model's max token limit by:
    1. Splitting long responses into smaller chunks
    2. Classifying each chunk separately
    3. Averaging the classification scores across chunks

    Args:
        cls: Classifier object containing the model and configuration
        responses: List of text responses to classify

    Returns:
        List of dictionaries, one per response, where each dict maps label names to scores
    """

    def mean_entries(values : list[dict]):
        """
        Calculate the mean score for each label across multiple classification results.

        Used to aggregate scores when a response was split into multiple chunks.

        Args:
            values: List of dicts, each mapping label -> score

        Returns:
            Single dict with averaged scores for each label
        """
        keys = values[0].keys()

        return {
            key: np.mean([v[key] for v in values]) for key in keys
        }

    def split_response(response : str, parts : int) -> list[str]:
        """
        Split a response into approximately equal parts based on word count.

        Args:
            response: Text string to split
            parts: Number of parts to split into

        Returns:
            List of text chunks
        """
        words = response.split(' ')
        split_size = math.ceil(len(words) / parts)
        return [' '.join(p) for p in batched(words, split_size)]

    tokenizer = cast(PreTrainedTokenizer, cls.pipe.tokenizer)

    # Prepare batch of entries for classification, splitting long responses as needed
    entries = []  # All text chunks to be classified
    indexes = []  # Track where each original response's chunks start in the entries list

    for res in responses:
        indexes.append(len(entries))
        ids = tokenizer.encode(res)

        max_tokens_nr = cls.max_length
        # If response fits within token limit, add it directly
        if len(ids) <= max_tokens_nr:
            entries.append(res)
            continue

        # Response is too long - split it into chunks
        splits_nr   = math.ceil(len(ids) / max_tokens_nr)
        new_entries = split_response(res, splits_nr)

        # Verify all chunks are within token limit; if not, increase split count
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        while any(map(lambda e : len(tokenizer.encode(e)) > max_tokens_nr, new_entries)):
            attempts += 1
            if attempts >= max_attempts:
                # If we can't split properly, just truncate to fit
                new_entries = [tokenizer.decode(ids[:max_tokens_nr-2])]  # -2 is because of special tokens
                break
            splits_nr += 1
            new_entries = split_response(res, splits_nr)

        entries.extend(new_entries)

    # Mark the end position for the last response
    indexes.append(len(entries))

    # Run classifier on all entries (chunks) at once
    results = [
        { entry['label'] : entry['score'] for entry in score } for score in tqdm(cast(list[dict], cls.pipe(entries)))
    ]

    # For each original response, average scores across its chunks
    scores = [
        mean_entries(results[indexes[i]: indexes[i+1]]) for i in range(len(indexes) - 1)
    ]

    # Ensure output has same length as input
    assert len(scores) == len(responses)
    return scores


def calculate_scores(classifiers : list[Classifier], responses : list[str]) -> list[list[dict]]:
    """
    Calculate classification scores for a list of responses using multiple classifiers.

    This function takes multiple classifiers and a list of text responses, runs each
    classifier on all responses, and returns the results in a structured format.

    Args:
        classifiers: List of Classifier objects to use for scoring
        responses: List of text strings to be classified

    Returns:
        A list of lists of dictionaries. The outer list has one entry per classifier,
        the inner list has one entry per response, and each dict contains the classifier
        name, raw scores, and normalized score.
    """

    def normalize_score(score : dict) -> float:
        """Extract the Portuguese (PT) score, preferring PT-PT variant if available."""
        base_score = score.get('PT-PT', None)
        return score['PT'] if base_score is None else base_score
        # return int(round(10 * (1 - base_score), 0))

    # Run each classifier on all responses and store results with classifier name
    scores = [
        {
            'classifier': cls.model_name.split('/')[-1],  # Extract short name from model path
            'scores' : classify_responses(cls, responses)
        }
        for cls in classifiers
    ]

    # Transform the structure: for each classifier, create a list of dicts where each dict
    # contains the full score breakdown and a normalized PT score for that response
    scores  = [
        [
            {
                entry['classifier']: val,  # Full score dict from classifier
                f'{entry['classifier']}_score': normalize_score(val)  # Normalized PT score
            } for val in entry['scores']
        ] for entry in scores
    ]

    # Ensure we have exactly 2 classifiers as expected by the downstream code
    assert len(scores) == 2

    return scores


def main():
    parser = argparse.ArgumentParser(description='Classify responses using variant identification models')
    parser.add_argument('base_dir', help='Base directory containing JSON files to process')
    parser.add_argument('--output-dir', help='Output directory for scored results (default: <base-dir>/class_scores)')

    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir if args.output_dir else path.join(base_dir, 'class_scores')
    makedirs(output_dir, exist_ok=True)
    
    classifiers = [
        Classifier("liaad/PtVId"), 
        Classifier("bastao/PeroVaz_PT-BR_Classifier")
    ]

    print(f"Loading files from '{base_dir}'")
    for filename in glob(f'{base_dir}/*.json'):
        print(f"Loading '{filename}'")
        is_pt_pt = filename.endswith('_pt-pt.json')
        is_pt_br = filename.endswith('_pt-br.json')
        base_name = path.basename(filename).rstrip('.json') + '_scored_v1.csv'
        out_filename = path.join(output_dir, base_name)

        with open(filename) as file:
            data = json.load(file)

        prompt_type = "normal"
        if is_pt_pt:
            prompt_type = "pt-pt"
        elif is_pt_br:
            prompt_type = "pt-br"

        results = []
        for row in data:
            turns = row.pop('turns')
            # Extract dataset if it exists in the row
            dataset = row.get('dataset', None)

            base_row = row.copy()
            base_row['prompt_type'] = prompt_type
            if dataset is not None:
                base_row['dataset'] = dataset

            results.extend(
               base_row | {
                'turn_nr': turn_nr,
               } | turn for turn_nr, turn in enumerate(turns)
            )

        scores = calculate_scores(classifiers, [ row['assistant'].split('</think>')[-1] for row in results ])
        responses = [
            row | score_1 | score_2 for row, score_1, score_2 in zip(results, *scores) 
        ]

        df = pd.DataFrame( data = responses )

        df.set_index('prompt_id', inplace=True)
        df.to_csv(out_filename)
        print(f"Storing results in '{out_filename}'")


if __name__ == '__main__': 
    main()
