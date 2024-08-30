import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from into_the_unknown.model.utils import (
    Predictor,
    create_data_loaders,
    get_embedding_size,
    plot_scatter,
    plot_training_curve,
)


class LinearPredictor(Predictor):
    def __init__(self, embedding_size: int, learning_rate: float = 0.001):
        super().__init__()
        self.linear = nn.Linear(embedding_size, 1)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        # Calculate Euclidean distance for each dimension
        euclidean_dist = torch.sqrt(torch.square(emb1 - emb2))
        return self.linear(euclidean_dist).squeeze()


class PredictorPipeline:
    def __init__(self, args, hparams):
        self.args = args
        self.hparams = hparams
        self.setup_gpu()
        pl.seed_everything(self.hparams["seed"])
        self.setup_output_directory()
        self.embedding_size = get_embedding_size(self.args.hdf_file)
        self.train_loader, self.val_loader, self.test_loader = (
            self.setup_data_loaders()
        )
        self.callbacks = self.setup_callbacks()
        self.trainer = self.setup_trainer()
        self.model = LinearPredictor(
            embedding_size=self.embedding_size,
            learning_rate=self.hparams["learning_rate"],
        )

    def setup_gpu(self):
        if torch.cuda.is_available():
            print("GPU available. Applying Tensor Core utilization.")
            torch.set_float32_matmul_precision("medium")

    def setup_output_directory(self):
        os.makedirs(self.args.output_dir, exist_ok=True)

    def setup_data_loaders(self):
        train_file = os.path.join(self.args.data_dir, "train.csv")
        val_file = os.path.join(self.args.data_dir, "val.csv")
        test_file = os.path.join(self.args.data_dir, "test.csv")
        return create_data_loaders(
            train_file,
            val_file,
            test_file,
            self.args.hdf_file,
            self.args.param_name,
            batch_size=self.hparams["batch_size"],
        )

    def setup_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.hparams["early_stopping_patience"],
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args.output_dir,
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        return [early_stop_callback, checkpoint_callback]

    def setup_trainer(self):
        return pl.Trainer(
            max_epochs=self.hparams["max_epochs"],
            callbacks=self.callbacks,
            accelerator="auto",
            devices="auto",
            enable_progress_bar=True,
            default_root_dir=self.args.output_dir,
        )

    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.val_loader)
        plot_training_curve(self.trainer, self.args.output_dir)

    def predict(self, data_loader):
        predictions = self.trainer.predict(
            dataloaders=data_loader, ckpt_path="best"
        )
        predictions = torch.cat(predictions).cpu().numpy()
        targets = torch.cat([batch[2] for batch in data_loader]).cpu().numpy()
        return predictions, targets

    @staticmethod
    def calculate_metrics(predictions, targets):
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        pearson_corr, _ = pearsonr(predictions, targets)
        spearman_corr, _ = spearmanr(predictions, targets)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Pearson Correlation": pearson_corr,
            "Spearman Correlation": spearman_corr,
        }

    @staticmethod
    def print_metrics(metrics, dataset_name="Test"):
        print(f"\nFinal Evaluation on {dataset_name} Set:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    def save_metrics(self, metrics, dataset_name="test"):
        metrics_file = os.path.join(
            self.args.output_dir, f"{dataset_name}_metrics.txt"
        )
        with open(metrics_file, "w") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        print(f"Metrics saved to {metrics_file}")

    def evaluate(self, data_loader, dataset_name="Test"):
        predictions, targets = self.predict(data_loader)
        metrics = self.calculate_metrics(predictions, targets)
        self.print_metrics(metrics, dataset_name)
        self.save_metrics(metrics, dataset_name.lower())

        if dataset_name.lower() == "test":
            plot_scatter(
                predictions,
                targets,
                metrics["Pearson Correlation"],
                metrics["R2"],
                self.args.output_dir,
            )

        return metrics

    def run(self):
        self.train()
        self.evaluate(self.val_loader, "Validation")
        self.evaluate(self.test_loader, "Test")


def main(args: argparse.Namespace) -> None:
    hparams = {
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
    }

    pipeline = PredictorPipeline(args, hparams)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDDT Predictor")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, val.csv, and test.csv files",
    )
    parser.add_argument(
        "--hdf_file", type=str, required=True, help="Path to the HDF file"
    )
    parser.add_argument(
        "--param_name",
        type=str,
        required=True,
        help="Name of the parameter column to predict in the CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files",
    )

    # Hyperparameters
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Patience for early stopping",
    )

    args = parser.parse_args()

    main(args)
