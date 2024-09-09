#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: Thu 24 Nov 2022 00:17:31
Description: Download protein language model from Hugging Face
Usage:       python download_protein_model.py

@author: tsenoner

Available Protein Language Models (PLMs) on Hugging Face:
- Rostlab/prot_t5_xl_half_uniref50-enc
- Rostlab/ProstT5_fp16
- facebook/esm2_t6_8M_UR50D
- facebook/esm2_t12_35M_UR50D
- facebook/esm2_t30_150M_UR50D
- facebook/esm2_t33_650M_UR50D
- facebook/esm2_t36_3B_UR50D
- facebook/esm2_t48_15B_UR50D
- ElnaggarLab/ankh-base
- ElnaggarLab/ankh-large
"""

import argparse
import os
from pathlib import Path

from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoModel, AutoTokenizer


def download_plm(model_name, output_dir):
    """
    Downloads a pre-trained language model from Hugging Face and saves it to the specified directory.

    Args:
    model_name (str): The name of the model to download.
    output_dir (Path): The directory where the model should be saved.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the transformers cache and huggingface hub cache to a subdirectory within output_dir
    cache_dir = output_dir / "transformers_cache"
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download the model and tokenizer
    print(f"Downloading model '{model_name}'...")
    if model_name.startswith("Rostlab"):
        model = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Save the model and tokenizer to the output directory
    print(f"Saving model and tokenizer to '{output_dir}'...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Clean up cache directory
    # print(f"Cleaning up cache directory '{cache_dir}'...")
    # for file in cache_dir.glob("*"):
    #     if file.is_file():
    #         file.unlink()
    #     elif file.is_dir():
    #         os.rmdir(file)

    print("Download and save complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a pre-trained language model from Hugging Face."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to download.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="The directory where the model should be saved.",
    )

    args = parser.parse_args()

    download_plm(args.model_name, args.out_dir)
