"""Utility functions for loading dataset files."""

import json
from typing import List, Dict


def load_conversations_from_json(json_path: str) -> Dict[str, List[str]]:
    """
    Load conversations from a JSON file.

    Args:
        json_path: Path to the JSON file containing conversations

    Returns:
        Dictionary mapping conversation IDs to lists of messages
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
