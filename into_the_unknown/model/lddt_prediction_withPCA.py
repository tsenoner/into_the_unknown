import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ProteinEmbeddingDataset(Dataset):
    def __init__(self, data, hdf_file, pca):
        self.data = data
        self.hdf_file = hdf_file
        self.pca = pca

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        with h5py.File(self.hdf_file, "r") as hdf:
            query_emb = self.pca.transform(hdf[row["query"]][:].reshape(1, -1))
            target_emb = self.pca.transform(
                hdf[row["target"]][:].reshape(1, -1)
            )

        query_emb = torch.tensor(query_emb.flatten(), dtype=torch.float32)
        target_emb = torch.tensor(target_emb.flatten(), dtype=torch.float32)
        lddt_score = torch.tensor(row["lddt"], dtype=torch.float32)

        return query_emb, target_emb, lddt_score



def create_data_loaders(csv_file, hdf_file, batch_size=32, pca_components=128):
    data = pd.read_csv(csv_file)

    # Perform train, validation, test split (60%, 20%, 20%)
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Fit PCA on training data
    print("Fitting PCA...")
    with h5py.File(hdf_file, 'r') as hdf:
        train_embeddings = []
        for protein in tqdm(train_data['query'].unique()):
            try:
                embedding = hdf[protein][:].flatten()
                train_embeddings.append(embedding)
            except KeyError:
                print(f"Warning: Protein {protein} not found in HDF file. Skipping.")

        train_embeddings = np.array(train_embeddings)

    if len(train_embeddings) == 0:
        raise ValueError("No valid embeddings found in the HDF file for the training data.")

    pca = PCA(n_components=pca_components)
    pca.fit(train_embeddings)
    print("PCA fitted.")

    # Filter out rows with missing proteins
    def filter_valid_proteins(df):
        with h5py.File(hdf_file, 'r') as hdf:
            valid_proteins = df[df['query'].isin(hdf.keys()) & df['target'].isin(hdf.keys())]
        return valid_proteins

    train_data = filter_valid_proteins(train_data)
    val_data = filter_valid_proteins(val_data)
    test_data = filter_valid_proteins(test_data)

    # Create datasets
    train_dataset = ProteinEmbeddingDataset(train_data, hdf_file, pca)
    val_dataset = ProteinEmbeddingDataset(val_data, hdf_file, pca)
    test_dataset = ProteinEmbeddingDataset(test_data, hdf_file, pca)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class LDDTPredictor(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=64):
        super(LDDTPredictor, self).__init__()
        self.embedding_size = embedding_size

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

    def forward(self, emb1, emb2):
        proc1 = self.individual_layers(emb1)
        proc2 = self.individual_layers(emb2)

        combined = torch.cat([proc1 + proc2, torch.abs(proc1 - proc2)], dim=1)

        lddt_score = self.combined_layers(combined)

        return lddt_score.squeeze()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    for query_emb, target_emb, lddt_scores in loader:
        query_emb, target_emb, lddt_scores = (
            query_emb.to(device),
            target_emb.to(device),
            lddt_scores.to(device),
        )

        optimizer.zero_grad()
        predictions = model(query_emb, target_emb)
        loss = criterion(predictions, lddt_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_predictions.extend(predictions.cpu().detach().numpy())
        all_targets.extend(lddt_scores.cpu().numpy())

    r2 = r2_score(all_targets, all_predictions)
    return total_loss / len(loader), r2


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for query_emb, target_emb, lddt_scores in loader:
            query_emb, target_emb, lddt_scores = (
                query_emb.to(device),
                target_emb.to(device),
                lddt_scores.to(device),
            )

            predictions = model(query_emb, target_emb)
            loss = criterion(predictions, lddt_scores)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(lddt_scores.cpu().numpy())

    mse = total_loss / len(loader)
    r2 = r2_score(all_targets, all_predictions)

    return mse, r2, all_predictions, all_targets


def plot_training_curve(
    train_losses, val_losses, train_r2s, val_r2s, output_dir
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(train_r2s, label="Training R²")
    ax2.plot(val_r2s, label="Validation R²")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R²")
    ax2.set_title("Training and Validation R²")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close()


def plot_scatter(predictions, targets, pearson_corr, r2, output_dir):
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


def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up data
    train_loader, val_loader, test_loader = create_data_loaders(
        args.csv_file, args.hdf_file, batch_size=32, pca_components=128
    )

    # Set up model
    model = LDDTPredictor(embedding_size=128).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    best_val_loss = float("inf")
    patience = 5
    no_improve_count = 0

    for epoch in range(num_epochs):
        train_loss, train_r2 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_r2, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation R²: {val_r2:.4f}")
        print()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best_model.pth"),
            )
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training completed!")

    # Plot training curve
    plot_training_curve(
        train_losses, val_losses, train_r2s, val_r2s, args.output_dir
    )

    # Load best model and evaluate on test set
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_model.pth"))
    )
    test_loss, test_r2, test_predictions, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    print("Final Evaluation on Test Set:")
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # Compute Pearson correlation for final scatter plot
    test_pearson, _ = pearsonr(test_predictions, test_targets)

    # Plot scatter plot
    plot_scatter(
        test_predictions, test_targets, test_pearson, test_r2, args.output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDDT Predictor")
    parser.add_argument(
        "-c", "--csv_file", type=str, required=True, help="Path to the CSV file"
    )
    parser.add_argument(
        "-H", "--hdf_file", type=str, required=True, help="Path to the HDF file"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    args = parser.parse_args()

    main(args)
