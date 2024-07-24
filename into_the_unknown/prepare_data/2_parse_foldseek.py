import argparse
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm


def process_foldseek(fs_result, fs_parsed, uid_set):
    print("Processing Foldseek results...")
    if not fs_parsed.is_file():
        df = pd.read_csv(fs_result, sep="\t")
        df = df[
            [
                "query",
                "target",
                "fident",
                "alnlen",
                "nident",
                "mismatch",
                "qcov",
                "tcov",
                "lddt",
                "rmsd",
                "alntmscore",
            ]
        ]
        uniprot_pattern = re.compile(r"AF-([A-Z0-9]+)-")
        get_uniprot_id = lambda f: uniprot_pattern.search(f).group(1)
        df["query"] = df["query"].apply(get_uniprot_id)
        df["target"] = df["target"].apply(get_uniprot_id)
        df = df.drop_duplicates().reset_index(drop=True)
        df = df[df["query"] != df["target"]].reset_index(drop=True)
        df.to_csv(fs_parsed, index=False)
    else:
        df = pd.read_csv(fs_parsed)
    return df.loc[
        (df["query"].isin(uid_set)) & (df["target"].isin(uid_set))
    ].reset_index(drop=True)


def extract_same_len_proteins(df, fasta, same_len_file, same_len_fasta):
    print("Extracting proteins with same length...")
    seq_lens = {}
    seqs = {}
    for header, seq in Fasta(fasta).items():
        seqs[header] = seq
        seq_lens[header] = len(seq)

    df["qlen"] = df["query"].map(seq_lens)
    df["tlen"] = df["target"].map(seq_lens)
    same_len = df.loc[df["qlen"] == df["tlen"]]
    same_len.to_csv(same_len_file, index=False)

    same_len_headers = set(same_len["query"]) | set(same_len["target"])
    with open(same_len_fasta, "w") as handle:
        for header in same_len_headers:
            handle.write(f">{header}\n{seqs[header]}\n")


def aa_comp(seq):
    total = len(seq)
    counts = Counter(seq)
    return {aa: count / total for aa, count in counts.items()}


def compute_aa_comp(df, df_fs):
    print("Computing amino acid composition...")
    comp = df.set_index("uid")["seq"].apply(aa_comp)
    comp_df = pd.json_normalize(comp).fillna(0)
    comp_df.index = comp.index
    q_comps = comp_df.loc[df_fs["query"]].reset_index(drop=True)
    t_comps = comp_df.loc[df_fs["target"]].reset_index(drop=True)
    df_fs["aa_comp_diff"] = np.linalg.norm(q_comps - t_comps, axis=1)
    return df_fs


def calc_hfsp(seq_id, ungapped_len):
    seq_id = seq_id * 100
    hfsp = np.where(
        ungapped_len <= 11,
        seq_id - 100,
        np.where(
            ungapped_len <= 450,
            seq_id
            - 770 * ungapped_len ** (-0.33 * (1 + np.exp(ungapped_len / 1000))),
            seq_id - 28.4,
        ),
    )
    return np.where(hfsp < 0, np.nan, hfsp)


def compute_hfsp(df):
    print("Computing HFSP scores...")
    df["ungapped_len"] = df["nident"] + df["mismatch"]
    df["hfsp"] = calc_hfsp(df["fident"].values, df["ungapped_len"].values)
    return df


def fetch_emb(h5_file, unique_ids):
    print(f"Fetching embeddings from {h5_file.name}...")
    emb = {}
    with h5py.File(h5_file, "r") as hdf5:
        for key in tqdm(unique_ids, desc="Fetching embeddings"):
            if key in hdf5:
                emb[key] = hdf5[key][:].flatten()
    return emb


def calc_dist(q_emb, t_embs):
    if q_emb is None:
        return np.array([np.nan] * len(t_embs))
    valid_idx = [i for i, emb in enumerate(t_embs) if emb is not None]
    valid_t_embs = np.array([emb for emb in t_embs if emb is not None])
    dist = np.array([np.nan] * len(t_embs))
    if valid_t_embs.size > 0:
        valid_dist = cdist(
            q_emb.reshape(1, -1), valid_t_embs, metric="euclidean"
        ).flatten()
        for idx, d in zip(valid_idx, valid_dist):
            dist[idx] = d
    return dist


def add_dist_to_df(
    df, h5_file, q_col, t_col, dist_col, pca_dist_col=None, n_comp=None
):
    print(f"Computing distances for {h5_file.name}...")
    unique_ids = set(df[q_col]).union(df[t_col])
    emb = fetch_emb(h5_file, unique_ids)

    if pca_dist_col and n_comp:
        print(f"Performing PCA with {n_comp} components...")
        emb_array = np.array([e for e in emb.values()])
        pca = PCA(n_components=n_comp).fit(emb_array)
        pca_emb = {
            h: pca.transform(e.reshape(1, -1))[0] for h, e in emb.items()
        }
        var = np.cumsum(pca.explained_variance_ratio_)
        print(
            f"Variance maintained with {n_comp} components: {var[-1] * 100:.2f}%"
        )

    grouped = df.groupby(q_col)

    for query, group in tqdm(grouped, desc="Compute distances"):
        q_emb = emb.get(query)
        t_ids = group[t_col].values
        t_embs = [emb.get(target) for target in t_ids]

        dist = calc_dist(q_emb, t_embs)
        df.loc[group.index, dist_col] = dist

        if pca_dist_col and n_comp:
            pca_q_emb = pca_emb.get(query)
            pca_t_embs = [pca_emb.get(target) for target in t_ids]
            pca_dist = calc_dist(pca_q_emb, pca_t_embs)
            df.loc[group.index, pca_dist_col] = pca_dist

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Process Foldseek results and compute distances"
    )
    parser.add_argument(
        "base_dir",
        help="Base directory containing all input files and where output files will be saved",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    in_csv = base / f"{base.name}.csv"
    fasta = base / f"{base.name}.fasta"
    fs_result = base / "foldseek" / "result.tsv"
    fs_parsed = base / "foldseek" / f"{base.name}_foldseek_parsed.csv"
    emb_dir = base / "embeddings"
    out_csv = base / f"{base.name}_final.csv"

    print(f"Loading input CSV: {in_csv}")
    df = pd.read_csv(in_csv)

    df_fs = process_foldseek(fs_result, fs_parsed, set(df["uid"]))

    same_len_file = fs_parsed.with_name(f"{fs_parsed.stem}_same_len_pairs.csv")
    same_len_fasta = same_len_file.with_suffix(".fasta")
    extract_same_len_proteins(df_fs, fasta, same_len_file, same_len_fasta)

    df_fs = compute_aa_comp(df, df_fs)
    df_fs = compute_hfsp(df_fs)

    for h5_file in emb_dir.glob("*.h5"):
        plm = h5_file.stem.replace(f"{base.name}_", "")
        dist_col = f"dist_{plm}"
        df_fs = add_dist_to_df(
            df_fs,
            h5_file,
            "query",
            "target",
            dist_col,
            pca_dist_col=f"pca_dist_{plm}",
            n_comp=128,
        )

    print("Optimizing dataframe memory usage...")
    float_cols = df_fs.select_dtypes(include=["float"]).columns.to_list()
    df_fs[float_cols] = df_fs[float_cols].astype(np.float32)

    print(f"Saving final results to: {out_csv}")
    df_fs.to_csv(out_csv, index=False)

    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
