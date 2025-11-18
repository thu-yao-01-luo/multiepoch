import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker

def load_data(file_path):
    """
    step,eval_loss,eval_PPL
    """
    data = pd.read_csv(file_path)
    step = data['step'].values
    eval_loss = data['eval_loss'].values
    return {
        'step': step,
        'eval_loss': eval_loss
    }

def find_nearest_index(array, value):
    """
    Find the index of the nearest value in the array to the given value.
    """
    array = np.asarray(array)
    # idx = (np.abs(array - value)).argmin()
    diff = array - value
    idx = np.where((diff > 1e-5), diff, np.inf).argmin()
    return idx

def smooth_data(y, window_length, polyorder):
    """Smooths 1D data using Savitzky-Golay filter."""
    try:
        return savgol_filter(y, window_length, polyorder)
    except Exception as e:
        print(f"Warning: Smoothing failed - {e}. Returning original data.")
        return y

if __name__ == '__main__':
    # Adjustable parameters
    font_size = 24
    line_weight = 1.5
    figure_size = (10, 7)
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2'] # Clear and readable color list
    n_epochs_label = "#epoch"
    grid_alpha = 0.5
    fresh_data_size_caption = "Fresh Data Size"
    epoch_caption = "#Epoch"
    ek_slope_stretch_factor = 0.2
    ek_slope_linestyle = '--'
    one_pass_color = '#7f7f7f'
    one_pass_linestyle = '-.'
    legend_fontsize = font_size - 2
    tick_fontsize = font_size - 2
    # title_fontsize = font_size + 4
    smoothing_window = 11 # Window length for smoothing (must be odd)
    smoothing_polyorder = 3 # Order of the polynomial to fit for smoothing
    shade_alpha = 0.2       # Transparency of the shaded area

    # Load data
    # me = ["0b2", "0b5", "0b6", "0b8", "1b", "2b"]
    me = ["0b2", "0b5", "0b8", "1b", "2b"]
    num_step_per_epoch = [400, 1000, 1600, 2000, 4000]
    op = ["200b", "200b_v2", "200b_v3"]
    label_map = {
        "0b2": "0.2B",
        "0b5": "0.5B",
        "0b6": "0.6B",
        "0b8": "0.8B",
        "1b": "1B",
        "2b": "2B",
    }
    data = {}
    loss_folder = "evals/"
    for m in me:
        file_path = os.path.join(loss_folder, f"{m}.csv")
        data[m] = load_data(file_path)
    for o in op:
        file_path = os.path.join(loss_folder, f"{o}.csv")
        data[o] = load_data(file_path)

    # compute E_k for each model
    # num_step_per_epoch = [400, 1000, 1200, 1600, 2000, 4000]
    for idx, model in enumerate(me):
        data[model]["E_k_step"] = []
        data[model]["E_k_loss"] = []
        for idx2, step in enumerate(data[model]["step"]):
            if True: # Keep the original logic for now
                data[model]["E_k_step"].append(step)
                data[model]["E_k_loss"].append(data[model]["eval_loss"][idx2])

    for idx, m in enumerate(me):
        data[m]["E_k_step"] = np.array(data[m]["E_k_step"])
        data[m]["E_k_loss"] = np.array(data[m]["E_k_loss"])
        data[m]["E_k"] = []
        for idx2, loss in enumerate(data[m]["E_k_loss"]):
            # equivalent_step_idx = find_nearest_index(data["200b"]["eval_loss"], loss)
            equivalent_step_idx = find_nearest_index(data["200b_v3"]["eval_loss"], loss)
            # if (data["200b"]["step"][equivalent_step_idx] <= data["200b_v2"]["step"][-1]):
            if (data["200b_v3"]["step"][equivalent_step_idx] <= data["200b_v2"]["step"][-1]):
                equivalent_step_idx = find_nearest_index(data["200b_v2"]["eval_loss"], loss)
                data[m]["E_k"].append(data["200b_v2"]["step"][equivalent_step_idx] / float(num_step_per_epoch[idx]))
            else:
                # data[m]["E_k"].append(data["200b"]["step"][equivalent_step_idx] / float(num_step_per_epoch[idx]))
                data[m]["E_k"].append(data["200b_v3"]["step"][equivalent_step_idx] / float(num_step_per_epoch[idx]))

    # --- Plot 1: E_k vs #epoch with smoothing and shading ---
    plt.figure(figsize=figure_size)
    for idx, m in enumerate(me):
        x = data[m]["E_k_step"] / num_step_per_epoch[idx]
        y_original = data[m]["E_k"]
        smoothed_y = smooth_data(y_original, smoothing_window, smoothing_polyorder)
        plt.plot(x, smoothed_y,
                 label=f"{label_map[m]}",
                 linewidth=line_weight, color=colors[idx % len(colors)])
        plt.fill_between(x, smoothed_y, y_original,
                         color=colors[idx % len(colors)], alpha=shade_alpha)

    # Add slope E_k = #epoch
    max_epoch = x.max() * ek_slope_stretch_factor
    plt.plot([0, max_epoch], [0, max_epoch], linestyle=ek_slope_linestyle,
                color=colors[-1], alpha=0.7,)
    plt.text(max_epoch * 0.01, max_epoch * 0.63, f"E(K,N)={n_epochs_label}",
                fontsize=legend_fontsize, color=colors[-1], alpha=0.7,
                ha='left', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor=colors[-1], facecolor='white', alpha=0.7))
                # label=f"$E_k$={n_epochs_label}") # Label only once

    plt.xlabel(n_epochs_label, fontsize=font_size)
    plt.ylabel(r"E(K,N)", fontsize=font_size)
    # plt.title(r"$E_k$ vs " + n_epochs_label, fontsize=title_fontsize)
    plt.grid(True, linestyle='-', alpha=grid_alpha)
    plt.legend(fontsize=legend_fontsize, loc='upper left', title=fresh_data_size_caption, title_fontsize=legend_fontsize, ncol=2)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig("E_k_smoothed_shaded.png")
    plt.savefig("E_k_smoothed_shaded.pdf")
    plt.show()
    plt.close()

    # --- Plot 2: Loss over Step ---
    # -- Plot 2.1---
    # --- Dummy Data and Setup (to make the code runnable) ---
    # In your actual code, you would load your 'data' dictionary here
    label_map = {
        "0b2": "200M", "0b5": "500M", "0b8": "800M", "1b": "1B", "2b": "2B"
    }
    # colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
    # colors = plt.get_cmap('tab20c')
    one_pass_color = '#e377c2'
    one_pass_linestyle = '--'
    line_weight = 2.5
    font_size = 20
    legend_fontsize = 16
    tick_fontsize = 16
    grid_alpha = 0.6
    figure_size = (10, 7)

    # -----------------------------------------------------------------
    # --- Improved Plotting Code ---
    # -----------------------------------------------------------------
    plt.figure(figsize=figure_size)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "legend.fontsize": legend_fontsize,
    })

    # Plot the baseline "200b_v3" curve first
    ref_steps = data["200b_v3"]["step"]
    ref_loss = data["200b_v3"]["eval_loss"]
    plt.plot(ref_steps, ref_loss,
            label="One Pass (200B)",
            linewidth=line_weight,
            color=one_pass_color,
            linestyle=one_pass_linestyle)

    # Loop through each model size to plot and annotate
    for idx, m in enumerate(me):
        model_color = colors[idx % len(colors)]
        model_steps = data[m]["step"]
        model_loss = data[m]["eval_loss"]
        if idx == 0 or idx == 2:
            continue
        # Plot the current model's loss curve
        plt.plot(model_steps, model_loss,
                label=label_map[m],
                linewidth=line_weight,
                color=model_color)

        # --- Find the Equivalent Step ---
        # 1. For each step of model 'm', find the equivalent step on the reference curve
        # that has the same loss value.
        equivalent_ref_steps = np.interp(model_loss, ref_loss[::-1], ref_steps[::-1])

        # 2. Calculate the ratio of the equivalent reference step to the model's own step.
        step_ratio = equivalent_ref_steps / model_steps

        # 3. Find the index where this ratio is closest to 0.99.
        # We search in the first half of the data to find the first crossing
        search_range = len(step_ratio) // 2
        target_idx = np.argmin(np.abs(step_ratio[:search_range] - 0.8))

        # --- Add Annotations ---
        if target_idx > 0: # Ensure a valid point was found
            # Get the coordinates of the target point on the model's curve
            target_step_m = model_steps[target_idx]
            target_loss_m = model_loss[target_idx]

            # Get the coordinates of the corresponding point on the reference curve
            target_step_ref = equivalent_ref_steps[target_idx]
            if idx == 0: 
                continue

            # 1. Mark the point on the model's curve with a scatter plot
            plt.scatter(target_step_m, target_loss_m, s=50, color=model_color, zorder=5)

            # 2. Add a dashed line connecting the two equivalent points
            plt.plot([target_step_m, target_step_ref], [target_loss_m, target_loss_m],
                    color=model_color, linestyle='--', linewidth=1.5, alpha=0.8)

            # 3. Add the text label
            epochs = target_step_m / num_step_per_epoch[idx]
            text_label = f"{label_map[m]} Tokens x {epochs:.1f} Epochs"
            plt.text(target_step_m + 800, target_loss_m - 0.015, text_label,
                    fontdict={"size": 18, "color": model_color})

    # --- Final Plot Formatting ---
    plt.xlabel("Step", fontsize=font_size)
    plt.ylabel("Validation Loss", fontsize=font_size)
    plt.grid(True, linestyle='-', alpha=grid_alpha)
    plt.legend(loc='upper right')
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xscale('log')
    # plt.ylim(2.2, 3.2) # Adjust ylim for better visibility
    plt.tight_layout()
    plt.savefig("loss_over_step_improved.pdf")
    plt.savefig("loss_over_step_improved.png")
    # plt.show()
    # plt.close()

    #--- Plot 3: E_k over N, different k ---
    k = [4, 16, 32, 64, 100]
    def epoch_step(x, k):
        if (len(x) * k) % 100 != 0:
            print(f"Warning: len(x) * k % 100 != 0, k={k}, len(x)={len(x)}")
        return round(len(x) * k / 100)-1
    plt.figure(figsize=figure_size)
    for m in me:
        print(len(data[m]["E_k"]))
    for idx, kk in enumerate(k):
        y = [data[m]["E_k"][epoch_step(data[m]["E_k"], kk)] for m in me]
        x = [float(label_map[m][:-1]) for m in me]

        plt.plot(x, y,
                 label=f"K={kk}", linestyle='--', marker='o',
                 linewidth=line_weight, color=colors[idx % len(colors)])
    plt.xlabel("Fresh Data Size (B)", fontsize=font_size)
    plt.ylabel(r"E(K,N)", fontsize=font_size)
    # plt.hlines(y=4, xmin=0, xmax=2, color='red', linestyle='-.', 
    plt.hlines(y=4.0, xmin=0, xmax=2, color='gray', linestyle='-.', linewidth=line_weight+1)
    plt.text(0.0, 4.5, r"E(K,N)=4", fontsize=legend_fontsize, color='gray', alpha=1.0,
                ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.7))
                # label=r"$E_k$=4", linewidth=line_weight+1)
    # plt.title(r"$E_k$ over Fresh Data Size", fontsize=title_fontsize)
    plt.grid(True, linestyle='-', alpha=grid_alpha)
    plt.legend(fontsize=legend_fontsize, loc='upper left', title=epoch_caption, title_fontsize=legend_fontsize+1)
    # x_loc = ticker.FixedLocator([0.2, 0.5, 0.6, 0.8, 1, 2])
    x_loc = ticker.FixedLocator([0.2, 0.5, 0.8, 1, 2])
    x_format = ticker.FuncFormatter(lambda x, pos: f"{x:.1f}")
    # plt.xticks(fontsize=tick_fontsize)
    plt.gca().xaxis.set_major_locator(x_loc)
    plt.gca().xaxis.set_major_formatter(x_format)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    # plt.xscale('log')
    plt.savefig("E_k_over_fresh_data_size.png")
    plt.savefig("E_k_over_fresh_data_size.pdf")
    plt.close()