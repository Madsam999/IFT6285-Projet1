import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.table import Table

# --- Configuration ---
# Set the name of the file to load
JSON_FILE_PATH = "../output/dev_BR-kNN_k10.json"
# Metrics we want to display on the bar chart (ideally metrics where higher is better)
CHART_METRICS = [
    "subset_accuracy",
    "f1_score_macro",
    "f1_score_micro",
    "jaccard_score_macro"
]

def load_metrics(filepath):
    """Loads metrics and hyperparameters from the specified JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: JSON file not found at {filepath}")
        return None, None
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics = data.get('metrics', {})
        hyperparameters = {}
        if "hyperparameters" in data and isinstance(data["hyperparameters"], dict):
            hyperparameters.update(data["hyperparameters"])
        if "hyperparameter_k" in data:
            hyperparameters["k"] = data["hyperparameter_k"]
        model_name = data.get('model_name', 'Unknown Model')
        
        return model_name, metrics, hyperparameters

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filepath}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return None, None

def plot_metrics_and_table(model_name, metrics, hyperparameters):
    """
    Creates a two-part visualization: 
    1. A bar chart of key performance metrics.
    2. A table summarizing all metrics and hyperparameters.
    """
    if not metrics:
        print("No metrics available for plotting.")
        return

    # --- Setup Figure and Subplots ---
    # Create a figure with two subplots: one for the chart and one for the table
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_chart = fig.add_subplot(gs[0, 0])
    
    fig.suptitle(f"{model_name} Performance Analysis", fontsize=16, fontweight='bold')
    
    # -----------------------------------------------------------------
    # 1. Bar Chart Visualization (Higher is Better Metrics)
    # -----------------------------------------------------------------
    
    # Filter and prepare data for the chart
    chart_data = {k: v for k, v in metrics.items() if k in CHART_METRICS}
    metric_names = list(chart_data.keys())
    metric_values = np.array(list(chart_data.values()))
    
    # Nicer labels for the plot
    nice_metric_names = [name.replace('_', ' ').title() for name in metric_names]
    
    bar_colors = plt.cm.Dark2(np.arange(len(metric_values))) # Use a color map for bars
    bars = ax_chart.bar(nice_metric_names, metric_values, color=bar_colors, width=0.6)
    
    ax_chart.set_ylabel("Score (0.0 to 1.0)")
    ax_chart.set_title("Key Multi-Label Performance Metrics")
    ax_chart.set_ylim(0, max(1.0, metric_values.max() * 1.1))
    ax_chart.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax_chart.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                      f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
                      
    # -----------------------------------------------------------------
    # 2. Metrics and Hyperparameter Table
    # -----------------------------------------------------------------

    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Load the data from the specified JSON file
    model_name, metrics, hyperparameters = load_metrics(JSON_FILE_PATH)
    print(model_name)
    print(metrics)
    print(hyperparameters)
    
    if metrics and hyperparameters:
        plot_metrics_and_table(model_name, metrics, hyperparameters)
    else:
        print(f"Script aborted due to errors loading data from {JSON_FILE_PATH}.")
