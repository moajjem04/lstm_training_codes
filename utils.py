# imports
import os
import numpy as np
import scipy.io as sio
import torch
import random
from scipy import stats

class Metrics:
    def __init__(self, name: str) -> None:
        self.name = name
        self.train = []
        self.val = []
        self.test = []

    def update(self, mode: str, value: float) -> None:
        if mode == "train":
            self.train.append(value)
        elif mode == "val":
            self.val.append(value)
        elif mode == "test":
            self.test.append(value)


def load_one_subject_data(filepath: str, binary_labels: bool = False) -> tuple[np.array, np.array, np.array]:
    # load data
    data = sio.loadmat(filepath)

    # extract data
    bvp = data["bvp"].astype(float)
    gsr = data["gsr"].astype(float)
    ratings = data["ratings"]    

    if binary_labels:
        # convert ratings to binary labels
        ratings = np.mean(ratings, axis=1)
        ratings = (ratings >= 2.1).astype(float)
    else:
        # convert ratings to one-hot encoding
        ratings = stats.mode(ratings, axis=1)[0].flatten(
            order="C"
        ).astype(float) - 1
        #print(ratings[:10])

    return bvp, gsr, ratings


def load_mat_data(
    root_dir: str, participant_list: list, binary: bool = False
) -> tuple[np.array, np.array, np.array]:

    bvp_list, gsr_list, ratings_list = [], [], []

    for participant in participant_list:
        filename = os.path.join(root_dir, f"P{participant:02}_all_bvp_gsr.mat")
        assert os.path.exists(filename), f"{filename} does not exist"

        bvp, gsr, ratings = load_one_subject_data(filename, binary_labels=binary)
        assert np.any(np.isnan(bvp)) == False, f"Subject {participant}'s bvp contains NaN"
        assert np.any(np.isnan(gsr)) == False, f"Subject {participant}'s gsr contains NaN"
        assert np.any(np.isnan(ratings)) == False, f"Subject {participant}'s ratings contains NaN"

        # append to list
        bvp_list.append(bvp)
        gsr_list.append(gsr)
        ratings_list.append(ratings)

    bvp_array = np.concatenate(bvp_list, axis=0)
    gsr_array = np.concatenate(gsr_list, axis=0)
    ratings_array = np.concatenate(ratings_list, axis=0)

    assert (
        bvp_array.shape[0] == gsr_array.shape[0] == ratings_array.shape[0]
    ), "Arrays have different lengths"
    #print(np.unique(ratings_array, return_counts=True))
    return bvp_array, gsr_array, ratings_array

def seed_everything(seed: int) -> None:
# Set a fixed seed for reproducibility
# seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False