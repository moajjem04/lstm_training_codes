# Imports
import os
import numpy as np
from rich.progress import Progress
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Custom Modules
from models import LSTMClassifier, CNNClassifier, MLPClassifier
from utils import Metrics, load_mat_data, seed_everything
from torch_utils import LSTM_Dataset, EarlyStopping


def probs_to_labels(probs: torch.Tensor) -> torch.Tensor:
    _, predicted = torch.max(probs, 1)
    return predicted


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        save_dir: str,
        es_patience: int,
        output_size: int,
        verbose: bool = True,
    ) -> None:
        # Initialize
        self.verbose = verbose
        self.epoch_start = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.ES = EarlyStopping(patience=es_patience, verbose=True, delta=0.0001)

        # Save Directory
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Running Stats
        self.metrics = {}
        self.metrics["loss"] = Metrics("loss")
        self.metrics["f1_score"] = Metrics("f1_score")
        self.metrics["accuracy"] = Metrics("accuracy")
        if output_size <= 2:
            self.f1_score = F1Score(task="binary", average="macro")
            self.accuracy = Accuracy(task="binary", average="macro")
        else:
            self.f1_score = F1Score(task="multiclass", num_classes=output_size, average="macro")
            self.accuracy = Accuracy(task="multiclass", num_classes=output_size, average="macro")

        # Change Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(self.device)
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.f1_score.to(self.device)
        self.accuracy.to(self.device)

    def _train_one_epoch(self, train_loader: DataLoader, progress) -> None:

        task2 = progress.add_task(
            "[blue]--->Training...", total=len(train_loader), visible=False
        )

        self.model.train()  # set model to training mode
        loss_list = []

        for i, (bvp, gsr, ratings) in enumerate(train_loader):
            # send to device
            bvp = bvp.to(self.device).float()
            gsr = gsr.to(self.device).float()
            ratings = ratings.to(self.device).long()
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            output = self.model(bvp, gsr)
            # print(output.dtype, ratings.dtype)
            loss = self.criterion(output, ratings)
            loss.backward()
            self.optimizer.step()
            # track stats
            output = probs_to_labels(output)
            self.f1_score.update(output, ratings)
            self.accuracy.update(output, ratings)
            loss_list.append(loss.item())
            # update progress
            progress.update(task2, advance=1)

        self.metrics["loss"].update("train", np.mean(loss_list))
        self.metrics["f1_score"].update(
            "train", self.f1_score.compute().detach().cpu().numpy()
        )
        self.metrics["accuracy"].update(
            "train", self.accuracy.compute().detach().cpu().numpy()
        )
        self.f1_score.reset()  # reset stats
        self.accuracy.reset()

    def _validate_one_epoch(self, val_loader: DataLoader, mode: str, progress) -> None:
        assert mode == "val" or mode == "test", "Mode should be 'val' or 'test'"

        _mode = "Validating" if mode == "val" else "Testing"
        task2 = progress.add_task(
            f"[blue]--->{_mode}...", total=len(val_loader), visible=False
        )

        loss_list = []
        self.model.eval()  # set model to evaluation mode

        with torch.no_grad():
            for i, (bvp, gsr, ratings) in enumerate(val_loader):
                bvp = bvp.to(self.device).float()
                gsr = gsr.to(self.device).float()
                ratings = ratings.to(self.device).long()

                output = self.model(bvp, gsr)
                # output = torch.squeeze(output)
                loss = self.criterion(output, ratings)
                # track stats
                output = probs_to_labels(output)
                self.f1_score.update(output, ratings)
                self.accuracy.update(output, ratings)
                loss_list.append(loss.item())
                # update progress
                progress.update(task2, advance=1)

        self.metrics["loss"].update(mode, np.mean(loss_list))
        self.metrics["f1_score"].update(
            mode, self.f1_score.compute().detach().cpu().numpy()
        )
        self.metrics["accuracy"].update(
            mode, self.accuracy.compute().detach().cpu().numpy()
        )

        self.f1_score.reset()  # reset stats
        self.accuracy.reset()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        print_every: int,
        resume: bool,
    ) -> None:
        if resume:
            self.load_state()
        self.epochs = epochs
        with Progress(transient=False) as progress:
            task1 = progress.add_task(
                "[red]Iterating through epochs...",
                total=epochs,
                completed=self.epoch_start,
            )

            for epoch in range(self.epoch_start, epochs):
                self.epoch = epoch
                # train one epoch
                self._train_one_epoch(train_loader, progress=progress)
                # validate one epoch
                self._validate_one_epoch(val_loader, mode="val", progress=progress)
                # test one epoch
                self._validate_one_epoch(test_loader, mode="test", progress=progress)
                # early stop
                self.ES(self.metrics["loss"].val[-1])  # compare with val loss

                if self.ES.early_stop:
                    if self.verbose:
                        self._print_stats()
                        print(
                            f"Early stopping at epoch {self.epoch:04d} with val loss {self.ES.best_score:.4f}"
                        )
                    break

                if self.ES.is_best:
                    # save model state
                    self.save_state()

                # print stats
                if epoch % print_every == 0 and self.verbose:
                    self._print_stats()
                # update progress
                progress.update(task1, advance=1)

        self._plot_stats()

    def inference(self, val_loader: DataLoader):
        self.load_state()
        with Progress(transient=False) as progress:
            self.model.eval()  # set model to evaluation mode
            task2 = progress.add_task(
                f"[blue]--->Inferring...", total=len(val_loader), visible=False
            )
            pred_list, true_list = [], []
            with torch.no_grad():
                for i, (bvp, gsr, ratings) in enumerate(val_loader):
                    bvp = bvp.to(self.device).float()
                    gsr = gsr.to(self.device).float()
                    ratings = ratings.to(self.device).long()
                    # pred
                    output = self.model(bvp, gsr)
                    output = probs_to_labels(output)
                    # convert to numpy
                    pred_list.extend(output.detach().cpu().numpy())
                    true_list.extend(ratings.detach().cpu().numpy())
                    # update progress
                    progress.update(task2, advance=1)

        # Generate classification report and confusion matrix
        class_report = classification_report(true_list, pred_list, digits=4, zero_division= 0.0)
        conf_matrix = confusion_matrix(true_list, pred_list)

        # Open a .txt file and write the results
        filepath = os.path.join(self.save_dir, "classification_report.txt")
        with open(filepath, 'w') as f:
            f.write("Classification Report:\n")
            f.write(class_report)
            f.write("\n\nConfusion Matrix:\n")
            np.savetxt(f, conf_matrix, fmt='%d')

        if self.verbose:
            print("Classification Report:\n")
            print(class_report)
            print("\nConfusion Matrix:\n")
            print(conf_matrix)
        acc = accuracy_score(true_list, pred_list)
        f1 = f1_score(true_list, pred_list, average = 'weighted')      

        return acc, f1

    def save_state(self) -> None:
        filepath = os.path.join(self.save_dir, "best_model_state.pt")

        state = {
            "epoch": self.epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(state, filepath)

    def load_state(self) -> None:
        filepath = os.path.join(self.save_dir, "best_model_state.pt")
        state = torch.load(filepath)
        self.epoch_start = state["epoch"] + 1
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])

    def _print_stats(self) -> None:
        print(f"Epoch: {self.epoch:04d}/{self.epochs:04d}")
        for key in self.metrics.keys():
            print(
                f"----Train {key}: {self.metrics[key].train[-1]:.4f}, Val {key}: {self.metrics[key].val[-1]:.4f}, Test {key}: {self.metrics[key].test[-1]:.4f}"
            )
        print(
            f"Max Memory Usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
        )

    def _plot_stats(self) -> None:
        for key in self.metrics.keys():
            filepath = os.path.join(self.save_dir, "plots", f"{key}.png")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.plot(self.metrics[key].train, label="train")
            plt.plot(self.metrics[key].val, label="val")
            plt.plot(self.metrics[key].test, label="test")
            plt.xlabel("Epoch")
            plt.ylabel(key)
            plt.legend()
            plt.savefig(filepath, dpi=300)
            plt.close()

def train_lstm(CONFIG: dict):
    ## folders
    root_dir = CONFIG["root_dir"]
    save_dir = CONFIG["save_dir"]
    SEED = CONFIG["SEED"]
    VERBOSE = CONFIG["VERBOSE"]

    ## model params
    MODEL = CONFIG["MODEL"]
    LENGTH = CONFIG["LENGTH"]
    IN_CH = CONFIG["IN_CH"]
    OUTPUT_SIZE = CONFIG["OUTPUT_SIZE"]
    HIDDEN_SIZE = CONFIG["HIDDEN_SIZE"]
    NUM_LAYERS = CONFIG["NUM_LAYERS"]
    DROPOUT = CONFIG["DROPOUT"]

    ## training params
    RESUME = CONFIG["RESUME"]
    EPOCHS = CONFIG["EPOCHS"]
    ES_PATIENCE = CONFIG["ES_PATIENCE"]
    PRINT_EVERY = CONFIG["PRINT_EVERY"]
    BS = CONFIG["BS"]
    LR = CONFIG["LR"]
    
    # Shuffle participants
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    participant_list = list(range(1, 31))
    rng.shuffle(participant_list)
    train_participant_list = participant_list[:20]
    val_participant_list = participant_list[20:25]
    test_participant_list = participant_list[25:]
    if VERBOSE:
        print(f"Subjects in train set: {train_participant_list}")
        print(f"Subjects in val set: {val_participant_list}")
        print(f"Subjects in test set: {test_participant_list}")
    
    # load data
    if OUTPUT_SIZE > 2:
        BINARY = False
    else:
        BINARY = True

    train_data = load_mat_data(root_dir, participant_list=train_participant_list, binary = BINARY)
    val_data = load_mat_data(root_dir, participant_list=val_participant_list, binary = BINARY)
    test_data = load_mat_data(root_dir, participant_list=test_participant_list, binary = BINARY)

    # create dataset
    train_ds = LSTM_Dataset(train_data[0], train_data[1], train_data[2])
    val_ds = LSTM_Dataset(val_data[0], val_data[1], val_data[2])
    test_ds = LSTM_Dataset(test_data[0], test_data[1], test_data[2])
    # create dataloader
    train_data = DataLoader(
        train_ds,
        batch_size=BS,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    val_data = DataLoader(
        val_ds, batch_size=BS, shuffle=False, pin_memory=True, num_workers=0
    )
    test_data = DataLoader(
        test_ds, batch_size=BS, shuffle=False, pin_memory=True, num_workers=0
    )
    # setup model
    if MODEL == "LSTM":
        model = LSTMClassifier(
            LENGTH, IN_CH, HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE, bidirectional=False, dropout=DROPOUT
        )
    elif MODEL == "MLP":
        model = MLPClassifier(
            LENGTH, IN_CH, HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE
        )
    elif MODEL == "CNN":
        model = CNNClassifier(
            LENGTH, IN_CH, HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE, dropout=DROPOUT
        )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    #criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    # train model
    trainer = Trainer(model, criterion, opt, save_dir=save_dir, es_patience=ES_PATIENCE, output_size=OUTPUT_SIZE, verbose=VERBOSE)
    trainer.train(
        train_data,
        val_data,
        test_data,
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
        resume=RESUME,
    )
    acc, f1 = trainer.inference(test_data)
    return acc, f1

if __name__ == "__main__":
    t = 4
    model = "CNN"
    main_config = {}
    main_config["root_dir"] = f"C:\\UofL - MSI\\DARPA\\mat_data\\user_wise_{t}s"
    main_config["save_dir"] = f"C:\\UofL - MSI\\DARPA\\Experiments\\EXP_{model}_{t}s"
    main_config["SEED"] = 123
    main_config["VERBOSE"] = True
    main_config["RESUME"] = False
    main_config["LENGTH"] = 64 * t
    main_config["IN_CH"] = 2
    main_config["OUTPUT_SIZE"] = 5
    main_config["EPOCHS"] = 100
    main_config["ES_PATIENCE"] = 50
    main_config["PRINT_EVERY"] = 10
    main_config["BS"] = 1024

    main_config["MODEL"] = model
    main_config["HIDDEN_SIZE"] = 64
    main_config["NUM_LAYERS"] = 4
    main_config["LR"] = 1e-4
    main_config["DROPOUT"] = 0.0

    # train
    _ = train_lstm(main_config)


# if __name__ == "__main__":
#     # CONFIG
#     ## folders
#     t = 4
#     root_dir = f"C:\\UofL - MSI\\DARPA\\mat_data\\user_wise_{t}s"
#     #root_dir = "C:\\UofL - MSI\\DARPA\\mat_data\\both_pup_user_wise"
#     save_dir = f"C:\\UofL - MSI\\DARPA\\Experiments\\EXP_CNN_{t}s"
#     SEED = 123
#     VERBOSE = True
#     ## model params
#     # INPUT_SIZE = 120 # bvp, gsr not 64, for lstm it is 2
#     LENGTH = 64*t
#     IN_CH = 2
#     OUTPUT_SIZE = 5
#     HIDDEN_SIZE = 64
#     NUM_LAYERS = 4
#     # training params
#     RESUME = False
#     EPOCHS = 1000
#     ES_PATIENCE = 100
#     PRINT_EVERY = 100
#     BS = 1024
#     LR = 1e-4
#     DROPOUT = 0.0

#     # Shuffle participants
#     seed_everything(SEED)
#     rng = np.random.default_rng(SEED)
#     participant_list = list(range(1, 31))
#     rng.shuffle(participant_list)
#     #print(participant_list[:20], participant_list[20:25], participant_list[25:])
#     train_participant_list = participant_list[:20]
#     val_participant_list = participant_list[20:25]
#     test_participant_list = participant_list[25:]
#     # load data
#     if OUTPUT_SIZE > 2:
#         BINARY = False
#     else:
#         BINARY = True
#     train_data = load_mat_data(root_dir, participant_list=train_participant_list, binary = BINARY)
#     val_data = load_mat_data(root_dir, participant_list=val_participant_list, binary = BINARY)
#     test_data = load_mat_data(root_dir, participant_list=test_participant_list, binary = BINARY)

#     # create dataset
#     train_ds = LSTM_Dataset(train_data[0], train_data[1], train_data[2])
#     val_ds = LSTM_Dataset(val_data[0], val_data[1], val_data[2])
#     test_ds = LSTM_Dataset(test_data[0], test_data[1], test_data[2])
#     # create dataloader
#     train_data = DataLoader(
#         train_ds,
#         batch_size=BS,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=0,
#         drop_last=True,
#     )
#     val_data = DataLoader(
#         val_ds, batch_size=BS, shuffle=False, pin_memory=True, num_workers=0
#     )
#     test_data = DataLoader(
#         test_ds, batch_size=BS, shuffle=False, pin_memory=True, num_workers=0
#     )
#     # setup model
#     model = LSTMClassifier(
#         LENGTH, IN_CH, HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE, bidirectional=False, dropout=DROPOUT
#     )
#     # model = MLPClassifier(
#     #     INPUT_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE
#     # )
#     model = LSTMClassifier(
#         LENGTH, IN_CH, HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE, dropout=DROPOUT
#     )


#     opt = torch.optim.Adam(model.parameters(), lr=LR)
#     #criterion = nn.BCELoss()
#     criterion = nn.CrossEntropyLoss()
#     # train model
#     trainer = Trainer(model, criterion, opt, save_dir=save_dir, es_patience=ES_PATIENCE, output_size=OUTPUT_SIZE, verbose=VERBOSE)
#     trainer.train(
#         train_data,
#         val_data,
#         test_data,
#         epochs=EPOCHS,
#         print_every=PRINT_EVERY,
#         resume=RESUME,
#     )
#     trainer.inference(test_data)