import os
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from torch.utils.data import DataLoader, Dataset


class H5PyDataset(Dataset):
    def __init__(self, data, file_path, param_name):
        self.data = data
        self.param_name = param_name
        self.file_path = file_path
        self.file = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r", swmr=True)

        row = self.data.iloc[idx]
        query_emb = torch.tensor(
            self.file[row["query"]][:].flatten(), dtype=torch.float32
        )
        target_emb = torch.tensor(
            self.file[row["target"]][:].flatten(), dtype=torch.float32
        )
        param_value = torch.tensor(row[self.param_name], dtype=torch.float32)

        return query_emb, target_emb, param_value

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class H5PyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __del__(self):
        self.dataset.close()


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.file = h5py.File(dataset.file_path, "r", swmr=True)


class Predictor(pl.LightningModule):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.individual_layers = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.criterion = nn.MSELoss()
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        proc1 = self.individual_layers(emb1)
        proc2 = self.individual_layers(emb2)
        combined = torch.cat([proc1 + proc2, torch.abs(proc1 - proc2)], dim=1)
        out = self.combined_layers(combined)
        return out.squeeze()

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.val_predictions.append(preds)
        self.val_targets.append(targets)
        return {"val_loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.test_predictions.append(preds)
        self.test_targets.append(targets)
        return {"test_loss": loss, "preds": preds, "targets": targets}

    def predict_step(
        self, batch: tuple, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        query_emb, target_emb, _ = batch
        predictions = self(query_emb, target_emb)
        return predictions

    def _common_step(self, batch: tuple, batch_idx: int) -> tuple:
        query_emb, target_emb, param_value = batch
        predictions = self(query_emb, target_emb)
        loss = self.criterion(predictions, param_value)
        return loss, predictions, param_value

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_predictions, self.val_targets, "val")
        self.val_predictions.clear()
        self.val_targets.clear()

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(
            self.test_predictions, self.test_targets, "test"
        )
        self.test_predictions.clear()
        self.test_targets.clear()

    def _log_epoch_metrics(self, predictions, targets, prefix):
        all_preds = torch.cat(predictions).cpu().numpy()
        all_targets = torch.cat(targets).cpu().numpy()
        r2 = r2_score(all_targets, all_preds)
        pearson_corr, _ = pearsonr(all_preds, all_targets)
        self.log(f"{prefix}_r2", r2, on_epoch=True, prog_bar=True)
        self.log(
            f"{prefix}_pearson", pearson_corr, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )


def create_data_loaders(
    train_file: str,
    val_file: str,
    test_file: str,
    hdf_file: str,
    param_name: str,
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def load_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, usecols=["query", "target", param_name])

    def filter_valid_proteins(df: pd.DataFrame) -> pd.DataFrame:
        start_row = len(df)
        with h5py.File(hdf_file, "r") as hdf:
            valid_proteins = df[
                df["query"].isin(hdf.keys()) & df["target"].isin(hdf.keys())
            ]
        filtered_rows = len(valid_proteins)
        removed_rows = start_row - filtered_rows
        print(f"Filtered {removed_rows} rows ({removed_rows/start_row:.2%}.")
        return valid_proteins

    train_data = filter_valid_proteins(load_data(train_file))
    val_data = filter_valid_proteins(load_data(val_file))
    test_data = filter_valid_proteins(load_data(test_file))

    train_dataset = H5PyDataset(train_data, hdf_file, param_name)
    val_dataset = H5PyDataset(val_data, hdf_file, param_name)
    test_dataset = H5PyDataset(test_data, hdf_file, param_name)

    train_loader = H5PyDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
    val_loader = H5PyDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
    test_loader = H5PyDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


def plot_training_curve(trainer: pl.Trainer, output_dir: str) -> None:
    event_acc = EventAccumulator(trainer.logger.log_dir)
    event_acc.Reload()
    scalars = event_acc.Scalars
    train_losses = [scalar.value for scalar in scalars("train_loss_epoch")]
    val_losses = [scalar.value for scalar in scalars("val_loss_epoch")]
    val_r2s = [scalar.value for scalar in scalars("val_r2")]
    val_pearsons = [scalar.value for scalar in scalars("val_pearson")]

    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(epochs, train_losses, label="Training Loss")
    ax1.plot(epochs, val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Metrics")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_r2s, label="Validation R²", color="#5fd35b", ls="--")
    ax2.plot(
        epochs,
        val_pearsons,
        label="Validation pearson",
        color="#357533",
        ls="--",
    )
    ax2.set_ylabel("Correlation metric")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close()

    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation R²: {val_r2s[-1]:.4f}")


def plot_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    pearson_corr: float,
    r2: float,
    output_dir: str,
) -> None:
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(
        f"True vs Predicted\nPearson Correlation: {pearson_corr:.4f}, R²: {r2:.4f}"
    )
    plt.text(
        0.05,
        0.95,
        f"Pearson Correlation: {pearson_corr:.4f}\nR²: {r2:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )
    plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
    plt.close()
