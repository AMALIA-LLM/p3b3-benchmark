#!/usr/bin/env python3
"""
Calculate diversity metrics for conversations in all_prompts.json
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import warnings

from ..utils.data_loading import load_conversations_from_json

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def load_conversations(json_path: str) -> List[List[str]]:
    """Load conversations from JSON file."""
    data = load_conversations_from_json(json_path)

    # Convert dictionary of conversations to list
    # Sort by conversation ID to maintain consistent order
    conversations = [data[key] for key in sorted(data.keys())]

    return conversations


def calculate_distinct_n(texts: List[str], n: int) -> float:
    """
    Calculate Distinct-n metric: ratio of unique n-grams to total n-grams.
    Higher values indicate more diversity.
    """
    all_ngrams = []

    for text in texts:
        tokens = word_tokenize(text.lower())
        text_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(text_ngrams)

    if len(all_ngrams) == 0:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)

    return unique_ngrams / total_ngrams


def calculate_entropy(texts: List[str], n: int = 1) -> float:
    """
    Calculate Shannon entropy of n-grams.
    Higher entropy indicates more diversity.
    """
    all_ngrams = []

    for text in texts:
        tokens = word_tokenize(text.lower())
        text_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(text_ngrams)

    if len(all_ngrams) == 0:
        return 0.0

    ngram_counts = Counter(all_ngrams)
    total = len(all_ngrams)

    entropy = 0.0
    for count in ngram_counts.values():
        prob = count / total
        entropy -= prob * np.log2(prob)

    return entropy


def calculate_lexical_diversity(texts: List[str]) -> float:
    """
    Calculate Type-Token Ratio (TTR): ratio of unique words to total words.
    """
    all_tokens = []

    for text in texts:
        tokens = word_tokenize(text.lower())
        all_tokens.extend(tokens)

    if len(all_tokens) == 0:
        return 0.0

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)

    return unique_tokens / total_tokens


def calculate_vocab_size(texts: List[str]) -> int:
    """Calculate total vocabulary size."""
    all_tokens = set()

    for text in texts:
        tokens = word_tokenize(text.lower())
        all_tokens.update(tokens)

    return len(all_tokens)


def calculate_conversation_diversity_metrics(conversations: List[List[str]]) -> Dict:
    """Calculate diversity metrics for all conversations."""

    # Flatten all messages
    all_messages = []
    for conv in conversations:
        all_messages.extend(conv)

    print(f"Total conversations: {len(conversations)}")
    print(f"Total messages: {len(all_messages)}")
    print()

    metrics = {}

    # Distinct-n metrics
    print("Calculating Distinct-n metrics...")
    for n in [1, 2, 3]:
        distinct_n = calculate_distinct_n(all_messages, n)
        metrics[f'distinct_{n}'] = distinct_n
        print(f"  Distinct-{n}: {distinct_n:.4f}")
    print()

    # Entropy metrics
    print("Calculating Entropy metrics...")
    for n in [1, 2, 3]:
        entropy = calculate_entropy(all_messages, n)
        metrics[f'entropy_{n}gram'] = entropy
        print(f"  Entropy ({n}-gram): {entropy:.4f}")
    print()

    # Lexical diversity
    print("Calculating Lexical Diversity...")
    ttr = calculate_lexical_diversity(all_messages)
    metrics['type_token_ratio'] = ttr
    print(f"  Type-Token Ratio (TTR): {ttr:.4f}")
    print()

    # Vocabulary size
    vocab_size = calculate_vocab_size(all_messages)
    metrics['vocabulary_size'] = vocab_size
    print(f"  Vocabulary Size: {vocab_size}")
    print()

    # Length statistics
    print("Calculating Length Statistics...")
    message_lengths = [len(word_tokenize(msg)) for msg in all_messages]
    metrics['avg_message_length'] = np.mean(message_lengths)
    metrics['std_message_length'] = np.std(message_lengths)
    metrics['min_message_length'] = np.min(message_lengths)
    metrics['max_message_length'] = np.max(message_lengths)
    print(f"  Avg message length (tokens): {metrics['avg_message_length']:.2f}")
    print(f"  Std message length (tokens): {metrics['std_message_length']:.2f}")
    print(f"  Min message length (tokens): {metrics['min_message_length']}")
    print(f"  Max message length (tokens): {metrics['max_message_length']}")
    print()

    # Conversation length statistics
    print("Calculating Conversation Statistics...")
    conv_lengths = [len(conv) for conv in conversations]
    metrics['avg_turns_per_conversation'] = np.mean(conv_lengths)
    metrics['std_turns_per_conversation'] = np.std(conv_lengths)
    metrics['min_turns'] = np.min(conv_lengths)
    metrics['max_turns'] = np.max(conv_lengths)
    print(f"  Avg turns per conversation: {metrics['avg_turns_per_conversation']:.2f}")
    print(f"  Std turns per conversation: {metrics['std_turns_per_conversation']:.2f}")
    print(f"  Min turns: {metrics['min_turns']}")
    print(f"  Max turns: {metrics['max_turns']}")

    return metrics


def calculate_per_conversation_metrics(conversations: List[List[str]]) -> pd.DataFrame:
    """Calculate metrics for each individual conversation."""

    results = []

    for idx, conv in enumerate(conversations):
        conv_id = f"p{idx}"

        result = {
            'conversation_id': conv_id,
            'num_turns': len(conv),
            'distinct_1': calculate_distinct_n(conv, 1),
            'distinct_2': calculate_distinct_n(conv, 2),
            'lexical_diversity': calculate_lexical_diversity(conv),
            'vocabulary_size': calculate_vocab_size(conv),
            'avg_message_length': np.mean([len(word_tokenize(msg)) for msg in conv]),
            'total_tokens': sum(len(word_tokenize(msg)) for msg in conv)
        }

        results.append(result)

    return pd.DataFrame(results)


def main():
    # Load conversations
    json_path = "resources/all_prompts.json"
    output_dir = Path("outputs/dataset_stats/")
    print(f"Loading conversations from {json_path}...")
    conversations = load_conversations(json_path)
    print()

    # Calculate overall diversity metrics
    print("="*60)
    print("OVERALL DIVERSITY METRICS")
    print("="*60)
    print()

    overall_metrics = calculate_conversation_diversity_metrics(conversations)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save overall metrics
    overall_df = pd.DataFrame([overall_metrics])
    output_path = output_dir / "diversity_metrics_overall.csv"
    overall_df.to_csv(output_path, index=False)
    print()
    print(f"Overall metrics saved to {output_path}")
    print()

    # Calculate per-conversation metrics
    print("="*60)
    print("PER-CONVERSATION METRICS")
    print("="*60)
    print()

    per_conv_df = calculate_per_conversation_metrics(conversations)

    # Display summary statistics
    print("Summary Statistics:")
    print(per_conv_df.describe())
    print()

    # Save per-conversation metrics
    output_path = output_dir / "diversity_metrics_per_conversation.csv"
    per_conv_df.to_csv(output_path, index=False)
    print(f"Per-conversation metrics saved to {output_path}")
    print()

    # Print some interesting findings
    print("="*60)
    print("KEY FINDINGS")
    print("="*60)
    print()
    print(f"Most diverse conversation (Distinct-1): {per_conv_df.loc[per_conv_df['distinct_1'].idxmax(), 'conversation_id']}")
    print(f"  Distinct-1 score: {per_conv_df['distinct_1'].max():.4f}")
    print()
    print(f"Least diverse conversation (Distinct-1): {per_conv_df.loc[per_conv_df['distinct_1'].idxmin(), 'conversation_id']}")
    print(f"  Distinct-1 score: {per_conv_df['distinct_1'].min():.4f}")
    print()
    print(f"Largest vocabulary: {per_conv_df.loc[per_conv_df['vocabulary_size'].idxmax(), 'conversation_id']}")
    print(f"  Vocabulary size: {per_conv_df['vocabulary_size'].max()}")
    print()
    print(f"Longest conversation: {per_conv_df.loc[per_conv_df['num_turns'].idxmax(), 'conversation_id']}")
    print(f"  Number of turns: {per_conv_df['num_turns'].max()}")


if __name__ == "__main__":
    main()


# example usage:
# python -m src.dataset_analysis.diversity