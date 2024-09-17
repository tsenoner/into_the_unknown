import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer

# Model checkpoints
Ankhs = [
    "ElnaggarLab/ankh-base",
    "ElnaggarLab/ankh-large",
]

ESMs = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
]

Rostlab = [
    "Rostlab/prot_t5_xl_uniref50",
    "Rostlab/ProstT5_fp16",
]


def process_fasta(fasta_path: Path, max_len=1022):
    """Remove sequences longer than max_len and save their identifiers to a file."""
    filtered_fasta_path = fasta_path.with_suffix(".filtered.fasta")
    long_sequences_path = fasta_path.with_suffix(".long_sequences.txt")

    with filtered_fasta_path.open(
        "w"
    ) as filtered_fasta, long_sequences_path.open("w") as long_sequences:
        for header, seq in Fasta(str(fasta_path)).items():
            if len(seq) > max_len:
                long_sequences.write(f"{header.split()[0]}\n")
            else:
                filtered_fasta.write(f">{header}\n{seq}\n")

    return filtered_fasta_path


def seq_preprocess(df, model_type="esm"):
    df["sequence"] = df["sequence"].str.replace("[UZO]", "X", regex=True)

    if model_type in "esm":
        return df
    elif model_type == "ankh":
        return df
    elif model_type == "pt":
        df["sequence"] = df.apply(lambda row: " ".join(row["sequence"]), axis=1)
        return df
    else:
        return None


def setup_model(checkpoint):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "esm" in checkpoint:
        mod_type = "esm"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = EsmModel.from_pretrained(checkpoint)
        model = model.to(device)
    elif "ankh" in checkpoint:
        mod_type = "ankh"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = T5EncoderModel.from_pretrained(checkpoint)
        model = model.to(device)
    else:
        mod_type = "pt"
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        model = T5EncoderModel.from_pretrained(
            checkpoint, torch_dtype=torch.float16
        )
        model = model.to(device)
        model = model.half()

    return model, tokenizer, mod_type


def read_fasta(file_path: Path):
    headers = []
    sequences = []
    fasta = Fasta(str(file_path))
    for seq in fasta:
        headers.append(seq.name)
        sequences.append(str(seq))
    return headers, sequences


def create_embedding(
    checkpoint,
    df,
    emb_type="per_prot",
    output_file: Path = Path("protein_embeddings.h5"),
):
    model, tokenizer, mod_type = setup_model(checkpoint)
    model.eval()
    df = seq_preprocess(df, mod_type)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_embedding(sequence, emb_type):
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            max_length=10_000,
            truncation=True,
            padding=True,
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.cpu().numpy()
        if emb_type == "per_res":
            # remove special tokens
            if mod_type in ["pt", "ankh"]:
                outputs = np.squeeze(outputs, axis=0)[:-1, :]
            elif mod_type == "esm":
                outputs = outputs[1:-1, :]
            return outputs
        elif emb_type == "per_prot":
            return outputs.mean(axis=1).flatten()
        else:
            raise ValueError("Input valid embedding type")

    # Open the HDF file in append mode
    with h5py.File(output_file, "a") as hdf:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sequence = row["sequence"]
            header = row["header"]

            # Check if the embedding already exists
            if header in hdf:
                continue

            embedding = compute_embedding(sequence, emb_type)
            hdf.create_dataset(name=header, data=embedding)

    # clean up gpu
    del model
    del tokenizer
    del df
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings from a FASTA file."
    )
    parser.add_argument(
        "fasta_file",
        type=Path,
        help="Path to the FASTA file containing protein sequences.",
    )
    parser.add_argument(
        "model_checkpoint",
        type=str,
        help="Pre-trained model checkpoint to use for embedding generation.",
    )
    parser.add_argument(
        "--emb_type",
        type=str,
        choices=["per_prot", "per_res"],
        default="per_prot",
        help="Type of embedding: per_prot or per_res (default: per_prot)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length. Sequences longer than this will be removed. (default: None)",
    )

    args = parser.parse_args()
    fasta_file = args.fasta_file

    if args.max_seq_len is not None:
        filtered_fasta_path = process_fasta(fasta_file, args.max_seq_len)
        headers, sequences = read_fasta(filtered_fasta_path)
    else:
        headers, sequences = read_fasta(fasta_file)

    df = pd.DataFrame({"header": headers, "sequence": sequences})

    output_file = fasta_file.with_name(
        f"{fasta_file.stem}_{Path(args.model_checkpoint).stem}.h5"
    )
    print(f"Embeddings out: {output_file}")
    create_embedding(
        args.model_checkpoint,
        df,
        emb_type=args.emb_type,
        output_file=output_file,
    )
    print(f"Embeddings saved to {output_file}")

    # Remove the temporary filtered FASTA file if it was created
    if args.max_seq_len is not None:
        filtered_fasta_path.unlink()
        print(f"Temporary filtered FASTA file removed: {filtered_fasta_path}")
