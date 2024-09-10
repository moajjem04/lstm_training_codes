import os
import scipy.io as sio
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    root_dir = "C:\\UofL - MSI\\DARPA\\mat_data"
    src_dir = "preprocessed"
    save_dir = "user_wise"
    participant = 1 # change to "P#""
    print(torch.cuda.is_available())

    # trial_list = [
    #     "Coffee_with error",
    #     "Coffee_without error",
    #     "Mugcake_with error",
    #     "Mugcake_without error",
    #     "Pinwheel_with error",
    #     "Pinwheel_without error",
    # ]  # 'Baseline' is removed

    # trial = trial_list[0] #iterate over trials
    # trial = trial.replace(" ", "_")

    # filename = os.path.join(root_dir, save_dir, f"P{participant:02}_{trial}_bvp_gsr.mat")
    # assert os.path.exists(filename)

    filename = os.path.join(root_dir, save_dir, f"P{participant:02}_all_bvp_gsr.mat")
    assert os.path.exists(filename)

    mat = sio.loadmat(filename)

    print(mat.keys())

    for key in mat.keys():
        if key[0] != '_':
            print(f"{key}: {mat[key].shape}")  

    print(mat['trial'][0])
    ratings = mat['ratings']
    plt.figure(figsize=(10, 5))
    plt.hist(ratings.ravel())
    plt.show()