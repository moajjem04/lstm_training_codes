'''
runner:
    - read mat files
    - send bvp and gsr to processor
    - receive windowed data
    - save all windowed data from all trials

processor:
    - window all data
    - filter if necessary
    - send windowed data to runner

'''

import os
import scipy.io as sio
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from rich.progress import Progress

class CombineAllTrials:
    def __init__(self, root_dir: str, src_dir: str, save_dir: str, participant: int, trial_list: list[str]) -> None:
        self.root_dir = root_dir
        self.src_dir = src_dir
        self.participant = participant
        self.trial_list = trial_list      
        self.save_dir = save_dir
        os.makedirs(os.path.join(self.root_dir, self.save_dir), exist_ok=True)

    def run(self, window_size = 64, exclude_length = 64*2, stride = 64, fs = 64):
        bvp_list, gsr_list, ratings_list, trial_list = [], [], [], []

        for trial in self.trial_list:
            try:
                self.load_single_trial_data(trial)
            except Exception as e:
                print(f"For Subject {self.participant}, Trial {trial} does not exist: {e}")
                continue
            # self.window_single_trial_data(window_size = 64, exclude_length = 64*2, fs = 64)
            self.window_single_trial_data(window_size, exclude_length, stride, fs)
            # append data
            bvp_list.append((self.bvp))
            gsr_list.append((self.gsr))
            ratings_list.append((self.ratings))
            trial_list.extend((self.trial))
            # clean up
            self.reset_single_trial_data()
        
        self.bvp = np.concatenate(bvp_list, axis = 0)
        self.gsr = np.concatenate(gsr_list, axis = 0)
        self.ratings = np.concatenate(ratings_list, axis = 0)
        self.trial = trial_list

        self.save_all_data()

    def load_single_trial_data(self, trial: str):
        filename = os.path.join(self.root_dir, self.src_dir, f"P{self.participant:02}_{trial}_bvp_gsr.mat")
        assert os.path.exists(filename), f"{filename} does not exist"
        # load data
        data = sio.loadmat(filename)
        # extract data
        self.bvp = data['bvp_value'].ravel()
        self.gsr = data['gsr_value'].ravel()
        self.ratings = data['bvp_ratings'].ravel()
        self.trial = trial

    def window_single_trial_data(self, window_size: int, stride :int, exclude_length: int, fs: int = None):
        bvp = self.bvp[exclude_length:]
        gsr = self.gsr[exclude_length:]
        ratings = self.ratings[exclude_length:]

        self.bvp = self._window_data(bvp, window_size, stride = stride)
        self.gsr = self._window_data(gsr, window_size, stride = stride)
        self.ratings = self._window_data(ratings, window_size, stride = stride)

        # make the same for trial
        self.trial = [self.trial for _ in range(self.bvp.shape[0])]

    def reset_single_trial_data(self):
        self.bvp = None
        self.gsr = None
        self.ratings = None
        self.trial = None
    def save_all_data(self):
        filename = os.path.join(self.root_dir, self.save_dir, f"P{self.participant:02}_all_bvp_gsr.mat")
        new_dict = {
            'bvp': self.bvp,
            'gsr': self.gsr,
            'ratings': self.ratings,
            'trial': self.trial
        }
        sio.savemat(filename, new_dict)

    def _window_data(self, data, window_size: int, stride: int = None) -> np.ndarray:
        if stride is None:
            stride = window_size
        windowed_data = sliding_window_view(data, window_shape = window_size)[::stride]

        return windowed_data


if __name__ == "__main__":
    root_dir = "C:\\UofL - MSI\\DARPA\\mat_data"
    # src_dir = "both_pup_trial_wise"
    # save_dir = "both_pup_user_wise"
    src_dir = "trial_wise"
    save_dir = "user_wise_8s"
    participant = 1 # change to "P#""

    trial_list = [
        "Coffee_with_error",
        "Coffee_without_error",
        "Mugcake_with_error",
        "Mugcake_without_error",
        "Pinwheel_with_error",
        "Pinwheel_without_error",
    ]  # 'Baseline' is removed

    FS = 64
    WINDOW_SIZE = 8 * FS
    EXCLUDE_LENGTH = 8 * FS
    STRIDE = 1 * FS
    

    combiner = CombineAllTrials(root_dir, src_dir, save_dir, participant, trial_list)
    #combiner.run(window_size=64, exclude_length=64*2, stride=64, fs=64)


    sub_list = list(range(1,30+1))

    with Progress() as progress:

        task1 = progress.add_task("[red]Iterating through subjects...", total=len(sub_list))

        for participant in sub_list:
            combiner = CombineAllTrials(root_dir, src_dir, save_dir, participant, trial_list)
            try:
                combiner.run(window_size=WINDOW_SIZE, exclude_length=EXCLUDE_LENGTH, stride=STRIDE, fs=FS)
            except Exception as e:
                print(f"Error processing data for P{participant:02}: {e}")
            progress.update(task1, advance=1)


