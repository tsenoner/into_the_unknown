import re
from collections import Counter
from pathlib import Path
from typing import Dict, Set

import h5py
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy.spatial.distance import cdist, euclidean
from sklearn.decomposition import PCA
from tqdm import tqdm


class ProteinAnalysis:
    def __init__(self, dataset_dir: str):
        self.BASE_DIR = Path(dataset_dir)
        self.BASE_NAME = self.BASE_DIR.stem
        self.FASTA_FILE = self.BASE_DIR / f"{self.BASE_NAME}_foldcomp.fasta"
        self.FOLDSEEK_DIR = self.BASE_DIR / "foldseek"
        self.FOLDSEEK_RESULT = self.FOLDSEEK_DIR / "result.tsv"
        self.FOLDSEEK_PARSED = self.FOLDSEEK_RESULT.with_suffix(".csv")
        self.FINAL_CSV = self.BASE_DIR / f"{self.BASE_NAME}_final.csv"
        self.H5_FILES = list((self.BASE_DIR / "embeddings").glob("*.h5"))

    @staticmethod
    def process_foldseek_results(
        foldseek_result: Path, foldseek_parsed: Path, uid_set: Set[str]
    ) -> pd.DataFrame:
        if not foldseek_parsed.is_file():
            df_fs = pd.read_csv(foldseek_result, sep="\t")
            df_fs = df_fs[
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
            get_uniprot_id = lambda file_name: uniprot_pattern.search(
                file_name
            ).group(1)

            df_fs["query"] = df_fs["query"].apply(get_uniprot_id)
            df_fs["target"] = df_fs["target"].apply(get_uniprot_id)

            df_fs = df_fs.drop_duplicates().reset_index(drop=True)
            df_fs = df_fs[df_fs["query"] != df_fs["target"]].reset_index(
                drop=True
            )
            df_fs.to_csv(foldseek_parsed, index=False)
        else:
            df_fs = pd.read_csv(foldseek_parsed)

        df_fs = df_fs.loc[
            (df_fs["query"].isin(uid_set)) & (df_fs["target"].isin(uid_set))
        ].reset_index(drop=True)
        return df_fs

    @staticmethod
    def create_fasta_file(df: pd.DataFrame, fasta_path: Path):
        records = df.apply(lambda row: f">{row['uid']}\n{row['seq']}\n", axis=1)
        with open(fasta_path, "w") as handle:
            handle.write("".join(records))

    def extract_same_length_proteins(self, df_fs: pd.DataFrame) -> None:
        df_fs = df_fs.copy()
        same_len_pairs_file = self.FOLDSEEK_PARSED.with_stem(
            f"{self.FOLDSEEK_PARSED.stem}_pairs"
        )
        same_len_pair_fasta = same_len_pairs_file.with_suffix(".fasta")

        seq_lens = {}
        seqs = {}
        for header, seq in Fasta(self.FASTA_FILE).items():
            seqs[header] = seq
            seq_lens[header] = len(seq)

        df_fs["qlen"] = df_fs["query"].map(seq_lens)
        df_fs["tlen"] = df_fs["target"].map(seq_lens)
        same_len_pairs = df_fs.loc[df_fs["qlen"] == df_fs["tlen"]]
        same_len_pairs.to_csv(same_len_pairs_file, index=False)

        same_len_headers = set(same_len_pairs["query"]) | set(
            same_len_pairs["target"]
        )
        with open(same_len_pair_fasta, "w") as handle:
            for header in same_len_headers:
                handle.write(f">{header}\n{seqs[header]}\n")

    @staticmethod
    def aa_composition(seq: str) -> Dict[str, float]:
        total_aa = len(seq)
        aa_counts = Counter(seq)
        return {aa: count / total_aa for aa, count in aa_counts.items()}

    @classmethod
    def compute_amino_acid_composition(
        cls, df: pd.DataFrame, df_fs: pd.DataFrame
    ) -> pd.DataFrame:
        seq_composition = df.set_index("uid")["seq"].apply(cls.aa_composition)
        seq_composition_df = pd.json_normalize(seq_composition).fillna(0)
        seq_composition_df.index = seq_composition.index

        query_comps = seq_composition_df.loc[df_fs["query"]].reset_index(
            drop=True
        )
        target_comps = seq_composition_df.loc[df_fs["target"]].reset_index(
            drop=True
        )

        df_fs["aa_comp_diff"] = np.linalg.norm(
            query_comps - target_comps, axis=1
        )
        return df_fs

    @staticmethod
    def calculate_hfsp(
        sequence_identity: np.ndarray, ungapped_alnlen: np.ndarray
    ) -> np.ndarray:
        sequence_identity = sequence_identity * 100
        hfsp = np.where(
            ungapped_alnlen <= 11,
            sequence_identity - 100,
            np.where(
                ungapped_alnlen <= 450,
                sequence_identity
                - 770
                * ungapped_alnlen
                ** (-0.33 * (1 + np.exp(ungapped_alnlen / 1000))),
                sequence_identity - 28.4,
            ),
        )
        hfsp = np.where(hfsp < 0, np.nan, hfsp)
        return hfsp

    @classmethod
    def compute_hfsp_scores(cls, df_fs: pd.DataFrame) -> pd.DataFrame:
        df_fs["ungapped_alnlen"] = df_fs["nident"] + df_fs["mismatch"]
        df_fs["hfsp"] = cls.calculate_hfsp(
            df_fs["fident"].values, df_fs["ungapped_alnlen"].values
        )
        return df_fs

    @staticmethod
    def fetch_embeddings(
        h5_file: str, unique_ids: Set[str]
    ) -> Dict[str, np.ndarray]:
        embeddings = {}
        with h5py.File(h5_file, "r") as hdf5:
            for key in tqdm(unique_ids, desc="Fetching embeddings"):
                if key not in hdf5:
                    continue
                embeddings[key] = hdf5[key][:].flatten()
        return embeddings

    @staticmethod
    def calculate_distances(
        query_emb: np.ndarray, target_embs: list
    ) -> np.ndarray:
        if query_emb is None:
            return np.array([np.nan] * len(target_embs))

        valid_indices = [
            i for i, emb in enumerate(target_embs) if emb is not None
        ]
        valid_target_embs = np.array(
            [emb for emb in target_embs if emb is not None]
        )

        distances = np.array([np.nan] * len(target_embs))

        if valid_target_embs.size > 0:
            valid_distances = cdist(
                query_emb.reshape(1, -1), valid_target_embs, metric="euclidean"
            ).flatten()
            for idx, dist in zip(valid_indices, valid_distances):
                distances[idx] = dist

        return distances

    @classmethod
    def add_distances_to_dataframe(
        cls,
        df: pd.DataFrame,
        h5_file: str,
        query_col: str,
        target_col: str,
        distance_col: str,
        pca_distance_col: str = None,
        n_components: int = None,
    ) -> pd.DataFrame:
        unique_ids = set(df[query_col]).union(df[target_col])
        embeddings = cls.fetch_embeddings(h5_file, unique_ids)
        # print(sum([1 for emb in embeddings if emb is None]))

        if pca_distance_col and n_components:
            embeddings_array = np.array([emb for emb in embeddings.values()])
            pca_model = PCA(n_components=n_components).fit(embeddings_array)
            pca_embeddings = {
                header: pca_model.transform(emb.reshape(1, -1))[0]
                for header, emb in embeddings.items()
            }
            explained_variance = np.cumsum(pca_model.explained_variance_ratio_)
            print(
                f"Variance maintained with {n_components} components: {explained_variance[-1] * 100:.2f}%"
            )

        # Group by the query to process all targets of each query in one go
        grouped = df.groupby(query_col)

        for query, group in tqdm(grouped, desc="Compute distances"):
            query_emb = embeddings.get(query)
            target_ids = group[target_col].values
            target_embs = [embeddings.get(target) for target in target_ids]

            distances = cls.calculate_distances(query_emb, target_embs)
            df.loc[group.index, distance_col] = distances

            if pca_distance_col and n_components:
                pca_query_emb = pca_embeddings.get(query)
                pca_target_embs = [
                    pca_embeddings.get(target) for target in target_ids
                ]
                pca_distances = cls.calculate_distances(
                    pca_query_emb, pca_target_embs
                )
                df.loc[group.index, pca_distance_col] = pca_distances

        # tqdm.pandas(desc="Calculating distances")
        # df[distance_col] = df.progress_apply(
        #     lambda row: cls.calculate_distance(
        #         embeddings.get(row[query_col]), embeddings.get(row[target_col])
        #     ),
        #     axis=1,
        # )

        # if pca_distance_col and n_components:
        #     tqdm.pandas(desc="Calculating PCA distances")
        #     df[pca_distance_col] = df.progress_apply(
        #         lambda row: cls.calculate_distance(
        #             pca_embeddings.get(row[query_col]),
        #             pca_embeddings.get(row[target_col]),
        #         ),
        #         axis=1,
        #     )

        return df

    def run_analysis(self, df):
        print("Process Foldseek results")
        df_fs = self.process_foldseek_results(
            self.FOLDSEEK_RESULT, self.FOLDSEEK_PARSED, set(df["uid"])
        )

        print("Save matching pairs with the same protein sequence length")
        self.extract_same_length_proteins(df_fs)

        print("Compute amino acid composition difference")
        df_fs = self.compute_amino_acid_composition(df, df_fs)

        print("Compute HFSP scores")
        df_fs = self.compute_hfsp_scores(df_fs)

        print("Compute pairwise distances")
        for h5_file in self.H5_FILES:
            print(h5_file)
            plm = h5_file.stem.replace(f"{self.BASE_NAME}_", "")
            distance_col = f"distance_{plm}"
            df_fs = self.add_distances_to_dataframe(
                df_fs,
                h5_file,
                "query",
                "target",
                distance_col,
                pca_distance_col=f"pca_distance_{plm}",
                n_components=128,
            )

        # Save final DataFrame
        float_columns = df_fs.select_dtypes(include=["float"]).columns.to_list()
        df_fs[float_columns] = df_fs[float_columns].astype(np.float32)
        print(
            f"After float32 conversion: {df_fs.memory_usage(deep=True).sum() / 1e6:.2f} MB"
        )
        df_fs.to_csv(self.FINAL_CSV, index=False)

        return df_fs


def main():
    base_dir = "data/s_pombe"
    protein_analysis = ProteinAnalysis(dataset_dir=base_dir)

    # Load the DataFrame from Part 1
    df = pd.read_csv(Path(base_dir) / f"{Path(base_dir).stem}_parsed.csv")

    protein_analysis.run_analysis(df)


if __name__ == "__main__":
    main()
