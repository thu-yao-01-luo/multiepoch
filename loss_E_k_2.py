import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker

# --- Data Loading and Processing Functions ---

def load_data(file_path):
    """Loads step and eval_loss from a CSV file."""
    data = pd.read_csv(file_path)
    return {'step': data['step'].values, 'eval_loss': data['eval_loss'].values}

def find_nearest_index(array, value):
    """Finds the index of the nearest value in an array."""
    array = np.asarray(array)
    diff = array - value
    idx = np.where((diff > 1e-5), diff, np.inf).argmin()
    return idx

def smooth_data(y, window_length, polyorder):
    """Smooths 1D data using Savitzky-Golay filter."""
    if len(y) < window_length:
        print(f"Warning: Data length ({len(y)}) is shorter than smoothing window ({window_length}). Returning original data.")
        return y
    return savgol_filter(y, window_length, polyorder)

def process_data(data, me, op, num_step_per_epoch):
    """Processes the loaded data to calculate E(K,N) values."""
    # Load raw data
    loss_folder = "evals/"
    for m in me:
        file_path = os.path.join(loss_folder, f"{m}.csv")
        if os.path.exists(file_path):
            data[m] = load_data(file_path)
    for o in op:
        file_path = os.path.join(loss_folder, f"{o}.csv")
        if os.path.exists(file_path):
            data[o] = load_data(file_path)
    
    # Check if baseline data for E(K,N) calculation is loaded
    if "200b_v3" not in data or "200b_v2" not in data:
        raise FileNotFoundError("Baseline data files (200b_v2.csv, 200b_v3.csv) not found in 'evals/' folder.")

    # Calculate E(K,N)
    for idx, model in enumerate(me):
        data[model]["E_k_step"] = np.array(data[model]["step"])
        data[model]["E_k_loss"] = np.array(data[model]["eval_loss"])
        data[model]["E_k"] = []
        for loss in data[model]["E_k_loss"]:
            equivalent_step_idx = find_nearest_index(data["200b_v3"]["eval_loss"], loss)
            if data["200b_v3"]["step"][equivalent_step_idx] <= data["200b_v2"]["step"][-1]:
                equivalent_step_idx_v2 = find_nearest_index(data["200b_v2"]["eval_loss"], loss)
                ek_val = data["200b_v2"]["step"][equivalent_step_idx_v2] / float(num_step_per_epoch[idx])
            else:
                ek_val = data["200b_v3"]["step"][equivalent_step_idx] / float(num_step_per_epoch[idx])
            data[model]["E_k"].append(ek_val)
    return data

# --- Plotting Functions ---

def plot_ek_vs_epoch(data, me, label_map, num_step_per_epoch, config):
    """Generates and saves the E(K,N) vs #Epoch plot (Left Plot)."""
    plt.figure(figsize=config['figure_size'])
    ax = plt.gca()
    
    # Set grid to be in the background
    ax.grid(True, linestyle='-', alpha=config['grid_alpha'], zorder=0)

    for idx, m in enumerate(me):
        x = data[m]["E_k_step"] / num_step_per_epoch[idx]
        y_original = data[m]["E_k"]
        smoothed_y = smooth_data(y_original, config['smoothing_window'], config['smoothing_polyorder'])
        
        color = config['colors'][idx % len(config['colors'])]
        
        # Plot smoothed curve
        ax.plot(x, smoothed_y, label=f"{label_map[m]}",
                linewidth=config['line_weight'], color=color, zorder=3)
        # Plot shaded area between original and smoothed data
        ax.fill_between(x, smoothed_y, y_original, color=color, 
                        alpha=config['shade_alpha'], zorder=2)

    # **IMPROVEMENT 2: Make the helper line more prominent**
    max_epoch = (data[me[-1]]["E_k_step"] / num_step_per_epoch[-1]).max() * config['ek_slope_stretch_factor']
    ax.plot([0, max_epoch], [0, max_epoch], 
            linestyle=config['ek_slope_linestyle'],
            color='black',  # Changed to black for prominence
            linewidth=config['line_weight'] - 0.5, # Slightly thinner than main lines
            alpha=1.0, # Fully opaque
            zorder=1) # Placed behind the data curves
            
    plt.text(max_epoch * 0.06, max_epoch * 0.8, f"E(K,N) = {config['n_epochs_label']}",
             fontsize=config['legend_fontsize'], color='black',
             ha='left', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8))

    ax.set_xlabel(config['n_epochs_label'], fontsize=config['font_size'])
    ax.set_ylabel(r"E(K,N)", fontsize=config['font_size'])
    ax.tick_params(axis='both', which='major', labelsize=config['tick_fontsize'])
    
    # **IMPROVEMENT 4: Move legend to the top**
    ax.legend(fontsize=config['legend_fontsize'], 
            #   title=config['fresh_data_size_caption'], 
              title_fontsize=config['legend_fontsize']-8, 
              ncol=5, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.15)) # (x, y) position, y > 1 is above the plot

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for top legend
    # plt.tight_layout()
    plt.savefig("E_k_smoothed_shaded.png")
    plt.savefig("E_k_smoothed_shaded.pdf")
    print("Generated E_k_smoothed_shaded plot.")
    plt.show()
    plt.close()

def plot_ek_vs_n(data, me, label_map, config):
    """Generates and saves the E(K,N) vs Fresh Data Size plot (Right Plot)."""
    plt.figure(figsize=config['figure_size'])
    ax = plt.gca()
    ax.grid(True, linestyle='-', alpha=config['grid_alpha'])

    # **IMPROVEMENT 3: Update K values**
    k_values = [4, 6, 8, 10, 16, 32, 64]

    def epoch_step(ek_data, k):
        # Calculate index corresponding to k% of the way through the epochs
        if not ek_data: return -1
        return min(round(len(ek_data) * k / 100) - 1, len(ek_data) - 1)

    for idx, k in enumerate(k_values):
        y = [data[m]["E_k"][epoch_step(data[m]["E_k"], k)] for m in me]
        x = [float(label_map[m][:-1]) for m in me]
        ax.plot(x, y, label=f"K={k}", linestyle='--', marker='o',
                linewidth=config['line_weight'], color=config['colors'][idx % len(config['colors'])])

    ax.set_xlabel("Fresh Data Size (B)", fontsize=config['font_size'])
    ax.set_ylabel(r"E(K,N)", fontsize=config['font_size'])
    
    # Helper line E(K,N)=4
    ax.hlines(y=4.0, xmin=0, xmax=2, color='gray', linestyle='-', linewidth=config['line_weight'])
    ax.text(0.05, 5.2, r"E(K,N)=4", fontsize=config['legend_fontsize'], color='gray',
            ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.7))

    # **IMPROVEMENT 4: Move legend to the top**
    ax.legend(fontsize=config['legend_fontsize']-8, 
            #   title=config['epoch_caption'], 
              title_fontsize=config['legend_fontsize'], 
              ncol=7, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.15))
              
    x_loc = ticker.FixedLocator([0.2, 0.5, 0.8, 1, 2])
    x_format = ticker.FuncFormatter(lambda x, pos: f"{x:.1f}")
    ax.xaxis.set_major_locator(x_loc)
    ax.xaxis.set_major_formatter(x_format)
    ax.tick_params(axis='both', which='major', labelsize=config['tick_fontsize'])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("E_k_over_fresh_data_size.png")
    plt.savefig("E_k_over_fresh_data_size.pdf")
    print("Generated E_k_over_fresh_data_size plot.")
    plt.close()

if __name__ == '__main__':
    # --- Centralized Configuration ---
    config = {
        'font_size': 24,
        'figure_size': (10, 7),
        # **IMPROVEMENT 1: Thicker lines**
        'line_weight': 2.5, 
        'colors': ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2'],
        'n_epochs_label': "#Epoch",
        'grid_alpha': 0.5,
        'fresh_data_size_caption': "Fresh Data Size",
        'epoch_caption': "#Epoch",
        'ek_slope_stretch_factor': 0.2,
        'ek_slope_linestyle': '--',
        'legend_fontsize': 22,
        'tick_fontsize': 22,
        'smoothing_window': 11,
        'smoothing_polyorder': 3,
        'shade_alpha': 0.15,
    }

    # --- Data Definition ---
    me = ["0b2", "0b5", "0b8", "1b", "2b"]
    op = ["200b_v2", "200b_v3"]
    num_step_per_epoch = [400, 1000, 1600, 2000, 4000]
    label_map = {"0b2": "0.2B", "0b5": "0.5B", "0b8": "0.8B", "1b": "1B", "2b": "2B"}
    
    # --- Main Execution ---
    try:
        # 1. Load and process data
        raw_data = {}
        processed_data = process_data(raw_data, me, op, num_step_per_epoch)
        
        # 2. Generate plots
        plot_ek_vs_epoch(processed_data, me, label_map, num_step_per_epoch, config)
        plot_ek_vs_n(processed_data, me, label_map, config)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the 'evals/' directory exists and contains all necessary CSV files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")