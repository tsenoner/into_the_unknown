import argparse
from pathlib import Path

import h5py
import pandas as pd
import torch
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer

# Model checkpoints
ESMs = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
]

Ankhs = ["ElnaggarLab/ankh-base", "ElnaggarLab/ankh-large"]
Rostlab = [
    "Rostlab/prot_t5_xl_uniref50",
    "Rostlab/ProstT5_fp16",
]


def seq_preprocess(df, model_type="esm"):
    df["sequence"] = df["sequence"].str.replace("[UZO]", "X", regex=True)

    if model_type in "esm":
        return df
    elif model_type == "ankh":
        #df["sequence"] = df["sequence"].apply(list)
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
        model = EsmModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        model = model.to(device)
        model = model.half()
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


def read_fasta(file_path):
    headers = []
    sequences = []
    fasta = Fasta(file_path)
    for seq in fasta:
        headers.append(seq.name)
        sequences.append(str(seq))
    return headers, sequences


def create_embedding(
    checkpoint, df, emb_type="per_prot", output_file="protein_embeddings.h5"
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
            add_special_tokens=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.cpu().numpy()
        if emb_type == "per_res":
            # remove special tokens
            if mod_type in ["pt", "ankh"]:
                outputs = outputs[:-1, :]
            elif mod_type == "esm":
                outputs = outputs[1:-1, :]
            return outputs
        elif emb_type == "per_prot":
            return outputs.mean(axis=1).flatten()
        else:
            raise ValueError("Input valid embedding type")

    with h5py.File(output_file, "w") as hdf:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sequence = row["sequence"]
            header = row["header"]
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
        type=str,
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
        help="Type of embedding to generate: per_prot or per_res.",
    )

    args = parser.parse_args()
    headers, sequences = read_fasta(args.fasta_file)
    df = pd.DataFrame({"header": headers, "sequence": sequences})

    fasta_file = Path(args.fasta_file)
    output_file = fasta_file.with_name(
        f"{fasta_file.stem}_{Path(args.model_checkpoint).stem}.h5"
    )
    print(f"Embeddings out: {output_file}")
    create_embedding(
        args.model_checkpoint,
        df,
        emb_type=args.emb_type,
        output_file=str(output_file),
    )
    print(f"Embeddings saved to {output_file}")
