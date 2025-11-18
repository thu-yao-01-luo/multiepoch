import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
    idx = (np.abs(array - value)).argmin()
    return idx

if __name__ == '__main__':
    # Load data
    me = ["0b2", "0b5", "0b6", "0b8", "1b", "2b"]
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
    num_step_per_epoch = [400, 1000, 1600, 2000, 4000]
    for idx, model in enumerate(me):
        data[model]["E_k_step"] = []
        data[model]["E_k_loss"] = []
        for idx2, step in enumerate(data[model]["step"]):
            # if step % num_step_per_epoch[idx] == 0:
            if True:
                data[model]["E_k_step"].append(step)
                data[model]["E_k_loss"].append(data[model]["eval_loss"][idx2])
    
    for idx, m in enumerate(me):
        data[m]["E_k_step"] = np.array(data[m]["E_k_step"])
        data[m]["E_k_loss"] = np.array(data[m]["E_k_loss"])
        data[m]["E_k"] = []
        for idx2, loss in enumerate(data[m]["E_k_loss"]):
            # equivalent_step_idx = find_nearest_index(data["200b"]["eval_loss"], loss)
            equivalent_step_idx = find_nearest_index(data["200b_v3"]["eval_loss"], loss)
            if (data["200b_v3"]["step"][equivalent_step_idx] <= data["200b_v2"]["step"][-1]):
                equivalent_step_idx = find_nearest_index(data["200b_v2"]["eval_loss"], loss)
                data[m]["E_k"].append(data["200b_v2"]["step"][equivalent_step_idx] / float(num_step_per_epoch[idx]))
            else:
                data[m]["E_k"].append(data["200b_v3"]["step"][equivalent_step_idx] / float(num_step_per_epoch[idx]))

    # plot E_k vs E_k_step / num_step_per_epoch
    plt.figure(figsize=(10, 6))
    for idx, m in enumerate(me):
        # print(f"model: {m}, step: {data[m]['E_k_step']}, E_k: {data[m]['E_k']}")
        plt.plot(data[m]["E_k_step"] / num_step_per_epoch[idx], data[m]["E_k"], label=label_map[m])
    plt.xlabel(r"n_epochs")
    plt.ylabel(r"$E_k$")
    plt.title(r"$E_k$ vs n_epochs")
    plt.legend()
    plt.savefig("E_k.png")
    plt.savefig("E_k.pdf")
    plt.show()
    plt.close()