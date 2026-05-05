# Dataset Analysis

Analyzes the dataset, focusing on diversity metrics and conversation structure statistics.

## Modules

### `diversity.py`

Calculates comprehensive diversity metrics for conversations in the dataset.

**Key Features:**
- **Distinct-n metrics**: Measures the ratio of unique n-grams (1,2,3) to total n-grams
- **Entropy metrics**: Calculates Shannon entropy for n-grams to quantify information diversity
- **Lexical diversity**: Computes Type-Token Ratio (TTR) for vocabulary richness
- **Vocabulary size**: Counts total unique words across conversations
- **Length statistics**: Analyzes message and conversation length distributions

**Metrics Generated:**

*Overall Dataset Metrics:*
- Distinct-1, Distinct-2, Distinct-3
- Entropy (1-gram, 2-gram, 3-gram)
- Type-Token Ratio (TTR)
- Vocabulary size
- Average/std/min/max message length (in tokens)
- Average/std/min/max turns per conversation

*Per-Conversation Metrics:*
- Number of turns
- Distinct-1 and Distinct-2 scores
- Lexical diversity
- Vocabulary size
- Average message length
- Total tokens

**Usage:**
```bash
python -m src.dataset_analysis.diversity
```

**Outputs:**
- `outputs/dataset_stats/diversity_metrics_overall.csv` - Overall dataset metrics
- `outputs/dataset_stats/diversity_metrics_per_conversation.csv` - Individual conversation metrics

### `turn_plots.py`

Creates visualizations of conversation turn distributions.

**Key Features:**
- Loads conversation data and counts turns per dialogue
- Generates bar plots showing frequency distribution of dialogue lengths

**Usage:**
```bash
python -m src.dataset_analysis.turn_plots
```

**Outputs:**
- `outputs/dataset_stats/number_of_turns_plot.pdf` - Turn distribution visualization

## Data Format

Both modules expect conversation data in JSON format at `resources/all_prompts.json`. The data should be structured as:
```json
{
  "conversation_id_1": ["message1", "message2", ...],
  "conversation_id_2": ["message1", "message2", ...],
  ...
}
```
