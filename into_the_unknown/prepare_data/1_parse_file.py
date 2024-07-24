import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import ijson
import pandas as pd
from tqdm import tqdm

class ProteinAnalysis:
    def __init__(self, dataset_dir: str, out_dir: str):
        self.BASE_DIR = Path(dataset_dir)
        self.BASE_NAME = self.BASE_DIR.stem
        self.JSON_PATH = self.BASE_DIR / f"{self.BASE_NAME}.json"
        self.CSV_JSON_PARSED = self.JSON_PATH.with_name(
            f"{self.JSON_PATH.stem}_parsed.csv"
        )
        self.FASTA_PATH = self.BASE_DIR / f"{self.BASE_NAME}.fasta"

    @staticmethod
    def extract_cross_references(
        entry: Dict, cross_references: List[str]
    ) -> Dict[str, str]:
        cross_references_data = {xref: [] for xref in cross_references}
        for xref in cross_references:
            xref_ids = [
                reference["id"]
                for reference in entry.get("uniProtKBCrossReferences", [])
                if reference["database"] == xref
            ]
            cross_references_data[xref] = ", ".join(xref_ids)
        return cross_references_data

    @classmethod
    def parse_entry(cls, entry):
        lineage = entry["organism"].get("lineage", [])
        lineage.append(entry["organism"].get("scientificName", ""))
        data = {
            "uid": entry.get("primaryAccession", ""),
            "taxon_id": entry["organism"].get("taxonId", ""),
            "lineage": ", ".join(lineage),
            "seq": entry["sequence"].get("value", ""),
        }

        cross_references = ["PDB", "Pfam", "AlphaFoldDB"]
        cross_references_data = cls.extract_cross_references(
            entry, cross_references
        )
        data.update(
            {f"{xref}": ids for xref, ids in cross_references_data.items()}
        )
        return data

    def parse_json_file(self) -> pd.DataFrame:
        if not self.CSV_JSON_PARSED.is_file():
            data = []
            with open(self.JSON_PATH, "r") as file:
                entries = ijson.items(file, "results.item")
                for entry in tqdm(entries, desc="Processing entries"):
                    data.append(self.parse_entry(entry))
            df = pd.DataFrame(data)
            df.to_csv(self.CSV_JSON_PARSED, index=False)
        else:
            df = pd.read_csv(self.CSV_JSON_PARSED)
        return df

    @staticmethod
    def create_fasta_file(df: pd.DataFrame, fasta_path: Path):
        records = df.apply(lambda row: f">{row['uid']}\n{row['seq']}\n", axis=1)
        with open(fasta_path, "w") as handle:
            handle.write("".join(records))

    def run_analysis(self):
        print("Parse JSON file")
        df = self.parse_json_file()

        print("Create FASTA file")
        self.create_fasta_file(df, self.FASTA_PATH)

def main():
    base_dir = "data/s_pombe"
    out_dir = "out"
    protein_analysis = ProteinAnalysis(dataset_dir=base_dir, out_dir=out_dir)
    protein_analysis.run_analysis()

if __name__ == "__main__":
    main()