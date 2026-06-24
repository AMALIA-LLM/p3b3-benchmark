import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from PIL import Image
import numpy as np

from ..utils.model_names import MODEL_RENAMES


# Plot styling constants
FIGURE_SIZE = (11, 8)
LINE_WIDTH = 2
MARKER_SIZE = 10
FONT_SIZE_LABEL = 16
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 16
LEGEND_COLS = 4
GRID_ALPHA = 0.3

# Provider configuration: prefix patterns, markers, colors, and images
PROVIDER_CONFIG = [
    {'name': 'BSC-LT', 'prefix': 'BSC-LT', 'marker': 'o', 'colors': ['#e74c3c', '#c0392b'], 'image': 'bsc.webp'},
    {'name': 'allenai', 'prefix': 'allenai', 'marker': 's', 'colors': ['#3498db', '#2980b9', '#5dade2'], 'image': 'olmo.webp'},
    {'name': 'utter-project', 'prefix': 'utter-project', 'marker': '^', 'colors': ['#9b59b6', '#8e44ad'], 'image': 'eurollm.webp'},
    {'name': 'swiss-ai', 'prefix': 'swiss-ai', 'marker': 'D', 'colors': ['#e67e22', '#d35400'], 'image': 'apertus.webp'},
    {'name': 'amalia-llm', 'prefix': 'amalia-llm', 'marker': 'v', 'colors': ['#1abc9c', '#16a085'], 'image': 'amalia.webp'},
    {'name': 'meta-llama', 'prefix': 'meta-llama', 'marker': 'p', 'colors': ['#2ecc71', '#27ae60'], 'image': 'meta.webp'},
    {'name': 'PORTULAN', 'prefix': 'PORTULAN', 'marker': '*', 'colors': ['#f39c12', '#e67e22'], 'image': 'gervasio.webp'},
    {'name': 'mistralai', 'prefix': 'mistralai', 'marker': 'h', 'colors': ['#34495e', '#2c3e50']},
    {'name': 'Qwen', 'prefix': 'Qwen', 'marker': 'H', 'colors': ['#e91e63', '#c2185b', '#f06292'], 'image': 'qwen.webp'},
    {'name': 'Polygl0t', 'prefix': 'Polygl0t', 'marker': '<', 'colors': ['#00bcd4']},
    {'name': 'google', 'prefix': 'google', 'marker': '>', 'colors': ['#4285f4', '#0f9d58', '#f4b400', '#db4437', '#673ab7'], 'image': 'google.webp'},
    {'name': 'openai', 'prefix': 'openai', 'marker': '+', 'colors': ['#10a37f']},
    {'name': 'maritaca-api', 'prefix': 'maritaca-api', 'marker': 'x', 'colors': ['#ff6f61', '#ff4757'], 'image': 'sabia.webp'},
    {'name': 'other', 'prefix': None, 'marker': 'o', 'colors': ['#95a5a6', '#7f8c8d']},
]

# Cache for loaded images
_image_cache = {}


def get_provider(model_name):
    """Extract provider from model name based on prefix matching."""
    for provider in PROVIDER_CONFIG:
        if provider['prefix'] and model_name.startswith(provider['prefix']):
            return provider['name']
    return 'other'


def get_provider_style(provider_name):
    """Get marker, colors, and image for a given provider."""
    for provider in PROVIDER_CONFIG:
        if provider['name'] == provider_name:
            return {
                'marker': provider['marker'],
                'colors': provider['colors'],
                'image': provider.get('image', None)
            }
    return {'marker': 'o', 'colors': ['#95a5a6', '#7f8c8d'], 'image': None}


def load_image_marker(image_filename, zoom=0.03):
    """
    Load and cache an image marker from the image_markers folder.

    Args:
        image_filename: Name of the image file
        zoom: Size factor for the image (default: 0.03)

    Returns:
        OffsetImage object or None if image not found
    """
    if image_filename in _image_cache:
        return _image_cache[image_filename]

    # Path to image_markers in project root (two levels up from src/analysis/)
    image_path = Path(__file__).parent.parent.parent / 'resources/image_markers' / image_filename

    try:
        img = Image.open(image_path)
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        imagebox = OffsetImage(np.array(img), zoom=zoom)
        _image_cache[image_filename] = imagebox
        return imagebox
    except Exception as e:
        print(f"Warning: Could not load image {image_filename}: {e}")
        _image_cache[image_filename] = None
        return None


class ImageHandler(HandlerBase):
    """Custom legend handler to display images and lines in legend."""

    def __init__(self, image_path, zoom=0.08):
        self.image_path = image_path
        self.zoom = zoom
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = []

        # First, draw the line from the original handle
        line = Line2D([xdescent, xdescent + width],
                     [ydescent + height/2., ydescent + height/2.],
                     color=orig_handle.get_color(),
                     linewidth=orig_handle.get_linewidth(),
                     transform=trans)
        artists.append(line)

        # Then, add the image marker in the center
        try:
            img = Image.open(self.image_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            imagebox = OffsetImage(np.array(img), zoom=self.zoom)
            imagebox.image.axes = legend.axes

            # Center the image on the line
            ab = AnnotationBbox(imagebox,
                              (xdescent + width/2., ydescent + height/2.),
                              xycoords=trans,
                              frameon=False,
                              pad=0,
                              bboxprops=dict(edgecolor='none', facecolor='none'))
            artists.append(ab)
        except Exception as e:
            print(f"Warning: Could not load legend image {self.image_path}: {e}")

        return artists


def plot_turn_progression(csv_path, show_plots=True, exclude_models=None, output_dir=None, max_turns=None, use_images=True):
    """
    Plot the progression of model scores across conversation turns.

    Args:
        csv_path: Path to CSV file with format:
                 model,turn_0,turn_1,turn_2,turn_3,turn_4,turn_5
        show_plots: Whether to display plots (default: True)
        exclude_models: List of model name patterns to exclude (None = include all models)
        output_dir: Directory to save plots (default: visualizations/)
        max_turns: Maximum number of turns to display (None = all turns)
        use_images: Whether to use image markers instead of standard markers (default: True)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Exclude models if specified
    if exclude_models:
        mask = ~df['model'].str.contains('|'.join(exclude_models), case=False, regex=True)
        df = df[mask]
        if len(df) == 0:
            print(f"Warning: All models were excluded by the filter: {exclude_models}")
            return
        print(f"Excluded models matching: {exclude_models}, {len(df)} models remaining")

    # Extract metadata from filename
    filename = Path(csv_path).stem

    # Parse variant (normal, pt-br, pt-pt)
    if '_normal_' in filename:
        variant = 'Normal'
    elif '_pt-br_' in filename:
        variant = 'pt-BR'
    elif '_pt-pt_' in filename:
        variant = 'pt-PT'
    else:
        variant = 'Unknown'

    # Parse scorer/classifier
    if 'PeroVaz' in filename:
        scorer = 'PeroVaz'
    elif 'PtVId' in filename:
        scorer = 'PtVId'
    elif 'llm' in filename.lower():
        scorer = 'LLM'
    else:
        scorer = 'Unknown'

    output_suffix = f'{variant.lower()}_{scorer.lower()}'

    # Set up output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent / 'visualizations'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set up the plot style
    plt.figure(figsize=FIGURE_SIZE)
    sns.set_style("whitegrid")

    # Get turn columns (all columns except 'model')
    turn_columns = [col for col in df.columns if col.startswith('turn_')]

    # Limit to max_turns if specified
    if max_turns is not None:
        turn_columns = turn_columns[:max_turns]
        print(f"Limiting to first {max_turns} turns")

    # Extract turn numbers from column names and create 1-indexed display labels
    turn_numbers = [int(col.split('_')[1]) for col in turn_columns]

    # Group models by provider
    provider_groups = {}
    for idx, row in df.iterrows():
        model_name = row['model']
        provider = get_provider(model_name)
        if provider not in provider_groups:
            provider_groups[provider] = []
        provider_groups[provider].append((idx, model_name, row))

    # Collect legend handles, labels, and handlers
    legend_handles = []
    legend_labels = []
    handler_map = {}

    # Plot each model's progression, grouped by provider
    for provider, models in provider_groups.items():
        style = get_provider_style(provider)
        marker = style['marker']
        colors = style['colors']
        image_filename = style.get('image', None)

        # Load image marker if available and use_images is True
        imagebox = None
        image_path = None
        if use_images and image_filename:
            # Use bigger zoom for specific providers
            zoom = 0.16 if provider == 'amalia-llm' else 0.10
            if provider == "meta-llama":
                zoom = 0.12
            imagebox = load_image_marker(image_filename, zoom=zoom)
            # Path to image_markers in project root (two levels up from src/analysis/)
            image_path = Path(__file__).parent.parent.parent / 'resources/image_markers' / image_filename

        for model_idx, (_, model_name, row) in enumerate(models):
            # Use pretty name if available, otherwise use original name
            display_name = MODEL_RENAMES.get(model_name, model_name)
            scores = [row[col] for col in turn_columns]

            # Cycle through provider-specific colors
            color = colors[model_idx % len(colors)]

            # Plot line
            plt.plot(turn_numbers, scores, linewidth=LINE_WIDTH, color=color)

            # Add image markers or fallback to standard markers
            if imagebox is not None:
                # Use image markers
                for turn_num, score in zip(turn_numbers, scores):
                    ab = AnnotationBbox(imagebox, (turn_num, score),
                                      frameon=False, pad=0)
                    plt.gca().add_artist(ab)
                # Create custom legend handle with image
                legend_handle = Line2D([0], [0], color=color, linewidth=LINE_WIDTH, label=display_name)
                # Use bigger zoom for specific providers in legend too
                legend_zoom = 0.14 if provider == 'amalia-llm' else 0.07
                if provider == 'meta-llama':
                    legend_zoom = 0.12
                handler_map[legend_handle] = ImageHandler(image_path, zoom=legend_zoom)
            else:
                # Fallback to standard markers
                plt.plot(turn_numbers, scores, marker=marker, color=color,
                        linestyle='', markersize=MARKER_SIZE)
                # Create legend handle with standard marker
                legend_handle = Line2D([0], [0], color=color, linewidth=LINE_WIDTH,
                                     marker=marker, markersize=MARKER_SIZE, label=display_name)

            legend_handles.append(legend_handle)
            legend_labels.append(display_name)

    # Customize the plot
    plt.xlabel('Turn Number', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    plt.ylabel(r'pt-BR  $\longleftarrow$  LLM Judge Score  $\longrightarrow$  pt-PT', fontsize=FONT_SIZE_LABEL, fontweight='bold')

    # plt.title(f'Alignment Across Turns - {variant} Prompt', fontsize=20, fontweight='bold', pad=20)
    plt.xticks(turn_numbers, fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)

    plt.legend(handles=legend_handles, labels=legend_labels,
              handler_map=handler_map,
              bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=FONT_SIZE_LEGEND, ncol=LEGEND_COLS)
    plt.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()

    # Save the plot
    version_suffix = "_v2" if use_images else ""
    output_path = output_dir / f"turn_progression_{output_suffix}{version_suffix}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    if show_plots:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot model performance progression across conversation turns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all models
  python turn_graphs.py results/z_classifier_scores/turn_level/aggregated_scores_by_turn_normal_all_prompts_PeroVaz_PT-BR_Classifier.csv

  # Exclude specific models
  python turn_graphs.py --exclude-models llama gemma AMALIA results/z_classifier_scores/turn_level/aggregated_scores_by_turn_normal_all_prompts_PeroVaz_PT-BR_Classifier.csv

  # Custom output directory
  python turn_graphs.py --output-dir ./plots results/z_llm_scores/turn_level/aggregated_scores_by_turn_pt-br_all_prompts_llm.csv

  # Save without displaying
  python turn_graphs.py --no-show --exclude-models Qwen results/z_classifier_scores/turn_level/aggregated_scores_by_turn_pt-pt_all_prompts_PtVId.csv

  # Show only first 3 turns
  python turn_graphs.py --max-turns 3 results/z_classifier_scores/turn_level/aggregated_scores_by_turn_normal_all_prompts_PeroVaz_PT-BR_Classifier.csv

  # Use standard markers instead of images
  python turn_graphs.py --no-images results/z_llm_scores/turn_level/aggregated_scores_by_turn_normal_all_prompts_llm.csv
        """
    )

    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to CSV file with turn-level scores (format: model,turn_0,turn_1,...)'
    )

    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (only save to files)'
    )

    parser.add_argument(
        '--exclude-models',
        type=str,
        nargs='+',
        # keeping the graph more clean with fewer models
        default=["maritaca-api-sabia-3.1", "openai-gpt-oss-20b", 'Qwen-Qwen3.5-9B', "google-gemma-3-27b-it",
                 "google-gemma-3-12b-it", "Polygl0t-Tucano2-qwen-3.7B-Instruct",
                 "PORTULAN-gervasio-70b-portuguese-ptpt-decoder", "amalia-llm-AMALIA-9B-50-1225-SFT",
                 "allenai-Olmo-3-7B-Instruct", "google-gemma-4-E4B-it", "swiss-ai-Apertus-70B-Instruct-2509",
                 "utter-project-EuroLLM-9B-Instruct-2512", 'mistralai-Ministral-3-14B-Instruct-2512',
                 'BSC-LT-salamandra-7b-instruct'],
        help='Exclude models by name patterns (space-separated).'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for plots (default: visualizations/ in same directory as CSV)'
    )

    parser.add_argument(
        '--max-turns',
        type=int,
        default=3,
        help='Maximum number of turns to display (default: 3)'
    )

    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Use standard markers instead of provider logo images'
    )

    args = parser.parse_args()

    # Validate file exists
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        parser.error(f"File not found: {args.csv_path}")

    # Generate plots
    plot_turn_progression(
        args.csv_path,
        show_plots=not args.no_show,
        exclude_models=args.exclude_models,
        output_dir=args.output_dir,
        max_turns=args.max_turns,
        use_images=not args.no_images
    )

"""
Example usage:

<data.csv> - this is the turn_level csv outputs from aggregation.py
                                                                                                                                                                           
# With image markers (default)                                                                                                                                           
python -m src.analysis.turn_analysis <data.csv> --max-turns 3

# With standard markers
python -m src.analysis.turn_analysis <data.csv> --max-turns 3 --no-images
"""
