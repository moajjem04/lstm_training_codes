# imports
import numpy as np
from torch.utils.data import Dataset

class LSTM_Dataset(Dataset):
    def __init__(self, bvp: np.array, gsr: np.array, ratings: np.array):
        self.bvp = np.expand_dims(bvp, axis=1)  # add channel dimension to bvp
        self.gsr = np.expand_dims(gsr, axis=1)  # add channel dimension to gsr
        self.ratings = np.expand_dims(ratings, axis=1)
        self.ratings = ratings
        #print(self.bvp.shape, self.gsr.shape, self.ratings.shape)

    def __len__(self):
        return self.bvp.shape[0]

    def __getitem__(self, idx):
        return self.bvp[idx], self.gsr[idx], self.ratings[idx]
    
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Initialize Early Stopping object.

        Parameters
        ----------
        patience : int, optional
            How long to wait after last time validation loss improved.
            Default: 5.
        verbose : bool, optional
            If True, prints a message for each validation loss improvement,
            thus indicating when early stopping is about to happen.
            Default: False.
        delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than `min_delta`, will count as no
            improvement. Default: 0.
        """

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.is_best = False

    def __call__(self, val_loss):
        score = val_loss  # maximize

        if self.best_score is None:
            self.best_score = score  # initialize
            self.is_best = True # this is the first time we are improving
        elif score > self.best_score - self.delta:
            self.counter += 1
            self.is_best = False # no improvement
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.is_best = True # improving
   