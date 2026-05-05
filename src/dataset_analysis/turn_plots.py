import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

from ..utils.data_loading import load_conversations_from_json


# Load conversations from JSON and calculate turn distribution
def load_turn_distribution(json_path: str = "resources/all_prompts.json"):
    """Load conversations and calculate the distribution of turns per dialogue."""
    data = load_conversations_from_json(json_path)

    # Count number of turns in each conversation
    turn_counts = [len(messages) for messages in data.values()]

    # Count frequency of each turn count
    turn_freq = Counter(turn_counts)

    # Sort by turn number and prepare data for plotting
    sorted_turns = sorted(turn_freq.items())
    turns = [str(t) for t, _ in sorted_turns]
    frequency = [f for _, f in sorted_turns]

    return turns, frequency


def create_turn_distribution_plot(json_path: str = "resources/all_prompts.json",
                                  output_path: str = "outputs/dataset_stats/number_of_turns_plot.pdf"):
    """Create and save a turn distribution plot from conversation data."""
    # Load data from JSON
    turns, frequency = load_turn_distribution(json_path)

    # Set style
    sns.set_style("whitegrid", {
        #'axes.spines.left': False,
        #'axes.spines.right': False,
        #'axes.spines.top': False
    })

    plt.figure(figsize=(10, 6))

    # Create the bar plot with a nice gradient palette
    palette = sns.color_palette("mako", len(turns))

    # decrease space between bars
    # Using a larger width for the bars will reduce the gap. (Original was 0.7)
    bars = plt.bar(turns, frequency, color=palette, edgecolor='white', linewidth=1, width=0.85)

    # Define new font sizes
    title_size = 20
    label_size = 18
    tick_size = 18
    data_label_size = 16

    # Title
    # plt.title('Distribution of Dialogue Lengths', fontsize=title_size, fontweight='bold', pad=20, color='#333333')

    # Axis labels with increased font size
    plt.xlabel('Number of Turns per Dialogue', fontsize=label_size, labelpad=12)
    plt.ylabel('Frequency', fontsize=label_size, labelpad=12)

    # Data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.7, # Adjusted text height for larger font
                 f'{int(height)}', ha='center', va='bottom',
                 fontsize=data_label_size, fontweight='bold', color='#444444')

    # Refine the grid and axes, increasing tick font size
    plt.yticks(fontsize=tick_size) # Increased tick font size
    plt.xticks(fontsize=tick_size) # Increased tick font size

    # Set y-limit dynamically with breathing room
    max_freq = max(frequency)
    plt.ylim(0, max_freq * 1.15)  # Add 15% breathing room

    # Check if output directory exists and create it if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    create_turn_distribution_plot()


# example usage:
# python -m src.dataset_analysis.turn_plots
