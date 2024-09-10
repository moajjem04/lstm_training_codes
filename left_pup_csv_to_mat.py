"""
The plan:
    - read bvp and gsr data
    - upsample gsr into bvp sampling rate
    - merge bvp and gsr into one .mat file
    - have subject id, trial id and labels
    - read three subject's data: one each for train, val, test

"""

import os
import pandas as pd
import numpy as np
import scipy.io as sio
from rich.progress import Progress

class ProcessData():
    def __init__(self, root_dir: str, src_dir: str, participant: int, trial: str, save_dir: str) -> None:
        self.root_dir = root_dir
        self.src_dir = src_dir
        self.participant = participant
        self.trial = trial
        
        self.new_trial = self.trial.replace(" ", "_") # replace space with _ for saving new file

        self.save_dir = save_dir
        os.makedirs(os.path.join(self.root_dir, self.save_dir), exist_ok=True)

    def read_data(self) -> None:

        """
        Read BVP and GSR data for a given participant and trial
        and return as two pandas DataFrames.

        Parameters
        ----------
        self : ProcessData

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            BVP and GSR data as two pandas DataFrames
        """
        
        self.data_dir = os.path.join(self.root_dir, self.src_dir, f"P{self.participant:02}", self.trial)
        #print(f"Reading data from {self.data_dir}")

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        bvp_file = os.path.join(self.data_dir, 'RightPup_time.csv') 
        gsr_file = os.path.join(self.data_dir, 'LeftPup_time.csv')

        if not os.path.exists(bvp_file) or not os.path.exists(gsr_file):
            raise FileNotFoundError(f"Either BVP_time.csv or GSR_time.csv does not exist in {self.data_dir}")

        try:
            bvp = pd.read_csv(bvp_file)
            gsr = pd.read_csv(gsr_file)
        except pd.errors.EmptyDataError:
            raise ValueError(f"Either BVP_time.csv or GSR_time.csv is empty in {self.data_dir}")

        # Assign bvp and gsr dfs to self
        self.bvp_df = bvp
        self.gsr_df = gsr
    
    def calculate_sampling_rate(self, verbose = 1) -> None:
        """
        Calculate the sampling rate of BVP and GSR data
        and store it in self.fs_bvp and self.fs_gsr
        """
        self.bvp_fs = 1 / self.bvp_df['time'].diff().mean()
        self.gsr_fs = 1 / self.gsr_df['time'].diff().mean()

        if verbose:
            print(f"BVP sampling rate: {self.bvp_fs:.2f}Hz")
            print(f"GSR sampling rate: {self.gsr_fs:.2f}Hz")
    def turn_to_numpy(self) -> None:
        """
        Convert BVP and GSR dataframes to numpy arrays
        and store them in self.bvp_value, self.bvp_t, self.bvp_ratings
        and self.gsr_value, self.gsr_t, self.gsr_ratings
        """
        
        self.bvp_value = self.bvp_df.value.to_numpy()
        self.bvp_t = self.bvp_df.time.to_numpy()
        self.bvp_ratings = self.bvp_df.ratings.to_numpy()

        self.gsr_value = self.gsr_df.value.to_numpy()
        self.gsr_t = self.gsr_df.time.to_numpy()
        self.gsr_ratings = self.gsr_df.ratings.to_numpy()     

    def upsample_one_data(self, target_t, actual_t, data) -> np.array:               
        """
        Interpolate data to match the time array target_t
        """
        
        # Interpolate        
        resampled_data = np.interp(target_t, actual_t, data)
        assert len(resampled_data) == len(target_t), "Interpolation failed"

        return resampled_data
    
    def upsample(self) -> None:
        """
        Upsample GSR data to match the sampling rate of BVP data
        and store the upsampled data in self.gsr_value, self.gsr_ratings, self.gsr_t
        """
        self.gsr_value = self.upsample_one_data(self.bvp_t, self.gsr_t, self.gsr_value)
        self.gsr_ratings = self.upsample_one_data(self.bvp_t, self.gsr_t, self.gsr_ratings)
        self.gsr_t = self.bvp_t
    
    def save_data(self) -> None:
        """
        Save BVP and GSR data as .mat files
        """
        new_dict = {
            'bvp_t': self.bvp_t,
            'bvp_value': self.bvp_value,
            'bvp_ratings': self.bvp_ratings,
            'gsr_t': self.gsr_t,
            'gsr_value': self.gsr_value,
            'gsr_ratings': self.gsr_ratings,
            'fs' : self.bvp_fs
        }

        filename = os.path.join(self.root_dir, self.save_dir, f"P{self.participant:02}_{self.new_trial}_bvp_gsr.mat")
        #print(f"\t-Saving data to {filename}")
        sio.savemat(filename, new_dict)

    def run(self) -> None:
        """
        Run the entire process
        """
        self.read_data()
        self.calculate_sampling_rate(verbose=0)
        self.turn_to_numpy()
        self.upsample()
        self.save_data()


   
if __name__ == "__main__":
    root_dir = "C:\\UofL - MSI\\DARPA"
    src_dir = "preprocessed"
    save_dir = "mat_data\\both_pup_trial_wise"
    participant = 1 # change to "P#""

    trial_list = [
        "Coffee_with error",
        "Coffee_without error",
        "Mugcake_with error",
        "Mugcake_without error",
        "Pinwheel_with error",
        "Pinwheel_without error",
    ]  # 'Baseline' is removed

    # trial = trial_list[0] #iterate over trials

    # # Process The data
    # processor = ProcessData(root_dir, src_dir,participant, trial, save_dir)
    # processor.run()

    sub_list = list(range(1,30+1))

    with Progress() as progress:

        task1 = progress.add_task("[red]Iterating through subjects...", total=len(sub_list))
        task2 = progress.add_task("[green]--Iterating through trials...", total=len(trial_list)*len(sub_list))

        for participant in sub_list:
            for trial in trial_list:
                processor = ProcessData(root_dir, src_dir, participant, trial, save_dir)
                try:
                    processor.run()
                except Exception as e:
                    print(f"Error processing data for P{participant:02} {trial}: {e}")
                progress.update(task2, advance=1)
                #break
            progress.update(task1, advance=1)
            #break

        # while not progress.finished:
        #     progress.update(task1, advance=0.5)
        #     progress.update(task2, advance=0.3)
        #     progress.update(task3, advance=0.9)
