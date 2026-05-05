# Analysis Module

This module contains scripts for aggregating model evaluation results and visualizing turn-level performance.

## Files

### `aggregation.py`

Main aggregation script that processes model evaluation results from the `results/` directory. It:

- Collects classifier scores and LLM judge scores from all model folders
- Groups results by prompt type (normal, pt-pt, pt-br) and dataset
- Generates both general and turn-level aggregated statistics
- Handles multiple evaluation runs by selecting only the latest timestamped files
- Computes averages with multiple filtering options (excluding invalid/non-pt responses)


**Output Directory Structure:**
```
results/
├── combined_comprehensive_scores.csv
├── z_classifier_scores/
│   ├── general/
│   │   └── aggregated_scores_{prompt_type}_{dataset}.csv
│   └── turn_level/
│       └── aggregated_scores_by_turn_{prompt_type}_{dataset}_{classifier}.csv
└── z_llm_scores/
    ├── general/
    │   └── aggregated_llm_scores_{prompt_type}.csv
    └── turn_level/
        └── aggregated_llm_scores_by_turn_{prompt_type}.csv
```

**Usage:**
```bash
python -m src.analysis.aggregation
```

### `turn_analysis.py`

Visualization script that plots showing model performance progression across conversation turns.

- Plots turn-level score progression for multiple models
- Groups models by provider with consistent color schemes
- Supports custom image markers for each provider (logo images)
- Automatically extracts metadata (variant, scorer) from CSV filenames


**Usage:**
```bash
# With image markers (default)
python -m src.analysis.turn_analysis <data.csv> --max-turns 3

# With standard markers
python -m src.analysis.turn_analysis <data.csv> --max-turns 3 --no-images
```

**Input Format:**
Output from `aggregation.py` turn-level CSV files, with the following structure:
CSV file with columns: `model,turn_0,turn_1,turn_2,...`

**Output:**
PDF plots saved to `{output_dir}/turn_progression_{variant}_{scorer}_v2.pdf`

## Notes

- Provider logo images should be placed in `resources/image_markers/`
- Model display names can be customized via `src.utils.model_names.MODEL_RENAMES`
