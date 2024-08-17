import argparse
import os
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from torch.utils.data import DataLoader, Dataset


class ProteinEmbeddingDataset(Dataset):
    def __init__(self, data: pd.DataFrame, hdf_file: str):
        self.data = data
        self.hdf_file = hdf_file
        # self.hdf_file = h5py.File(hdf_file, "r")

    # def __del__(self):
    #     self.hdf_file.close()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]

        with h5py.File(self.hdf_file, "r") as hdf:
            query_emb = torch.tensor(
                hdf[row["query"]][:].flatten(), dtype=torch.float32
            )
            target_emb = torch.tensor(
                hdf[row["target"]][:].flatten(), dtype=torch.float32
            )

        lddt_score = torch.tensor(row["lddt"], dtype=torch.float32)

        return query_emb, target_emb, lddt_score

    # def __getitem__(
    #     self, idx: int
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     row = self.data.iloc[idx]
    #     query_emb = torch.tensor(
    #         self.hdf_file[row["query"]][:].flatten(), dtype=torch.float32
    #     )
    #     target_emb = torch.tensor(
    #         self.hdf_file[row["target"]][:].flatten(), dtype=torch.float32
    #     )
    #     lddt_score = torch.tensor(row["lddt"], dtype=torch.float32)
    #     return query_emb, target_emb, lddt_score


class LDDTPredictor(pl.LightningModule):
    def __init__(
        self,
        embedding_size: int = 1024,
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

        lddt_score = self.combined_layers(combined)

        return lddt_score.squeeze()

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
        query_emb, target_emb, lddt_scores = batch
        predictions = self(query_emb, target_emb)
        loss = self.criterion(predictions, lddt_scores)
        return loss, predictions, lddt_scores

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
    csv_file: str, hdf_file: str, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data = pd.read_csv(csv_file)

    train_data, temp_data = train_test_split(
        data, test_size=0.4, random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42
    )

    def filter_valid_proteins(df: pd.DataFrame) -> pd.DataFrame:
        with h5py.File(hdf_file, "r") as hdf:
            valid_proteins = df[
                df["query"].isin(hdf.keys()) & df["target"].isin(hdf.keys())
            ]
        return valid_proteins

    train_data = filter_valid_proteins(train_data)
    val_data = filter_valid_proteins(val_data)
    test_data = filter_valid_proteins(test_data)

    train_dataset = ProteinEmbeddingDataset(train_data, hdf_file)
    val_dataset = ProteinEmbeddingDataset(val_data, hdf_file)
    test_dataset = ProteinEmbeddingDataset(test_data, hdf_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


def plot_training_curve(trainer: pl.Trainer, output_dir: str) -> None:
    # Get the logged metrics history
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

    # add correlation results
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

    # Print final metrics
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
    plt.plot([0, 1], [0, 1], "r--")  # Diagonal line
    plt.xlabel("True lDDT")
    plt.ylabel("Predicted lDDT")
    plt.title(
        f"True vs Predicted lDDT\nPearson Correlation: {pearson_corr:.4f}, R²: {r2:.4f}"
    )
    plt.text(
        0.05,
        0.95,
        f"Pearson Correlation: {pearson_corr:.4f}\nR²: {r2:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )
    plt.savefig(os.path.join(output_dir, "lddt_scatter_plot.png"))
    plt.close()


def main(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        print("GPU available. Applying Tensor Core utilization recommendation.")
        torch.set_float32_matmul_precision("medium")

    pl.seed_everything(42)

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = create_data_loaders(
        args.csv_file, args.hdf_file, batch_size=128
    )

    # --- prepare training ---
    refresh_rate = len(train_loader) % 10
    progress_bar = TQDMProgressBar(refresh_rate=refresh_rate, leave=True)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=3, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[early_stop_callback, checkpoint_callback, progress_bar],
        accelerator="auto",
        devices="auto",
        # enable_progress_bar=False,
        default_root_dir=args.output_dir,
    )

    model = LDDTPredictor(embedding_size=1024, learning_rate=0.001)
    trainer.fit(model, train_loader, val_loader)

    plot_training_curve(trainer, args.output_dir)

    test_results = trainer.test(dataloaders=test_loader, ckpt_path="best")

    test_predictions = trainer.predict(
        dataloaders=test_loader, ckpt_path="best"
    )
    test_predictions = torch.cat(test_predictions).cpu().numpy()
    test_targets = torch.cat([batch[2] for batch in test_loader]).cpu().numpy()

    test_r2 = r2_score(test_targets, test_predictions)
    test_pearson, _ = pearsonr(test_predictions, test_targets)

    plot_scatter(
        test_predictions, test_targets, test_pearson, test_r2, args.output_dir
    )

    print("Final Evaluation on Test Set:")
    print(f"Test Loss (MSE): {test_results[0]['test_loss_epoch']:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test Pearson Correlation: {test_pearson:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDDT Predictor")
    parser.add_argument(
        "-c", "--csv_file", type=str, required=True, help="Path to the CSV file"
    )
    parser.add_argument(
        "-H", "--hdf_file", type=str, required=True, help="Path to the HDF file"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    args = parser.parse_args()

    main(args)
