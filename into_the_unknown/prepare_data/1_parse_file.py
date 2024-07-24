import argparse
from pathlib import Path

import ijson
import pandas as pd
from tqdm import tqdm


def extract_cross_references(entry, cross_references):
    return {
        xref: ", ".join(
            [
                reference["id"]
                for reference in entry.get("uniProtKBCrossReferences", [])
                if reference["database"] == xref
            ]
        )
        for xref in cross_references
    }


def parse_entry(entry):
    lineage = entry["organism"].get("lineage", [])
    lineage.append(entry["organism"].get("scientificName", ""))
    data = {
        "uid": entry.get("primaryAccession", ""),
        "taxon_id": entry["organism"].get("taxonId", ""),
        "lineage": ", ".join(lineage),
        "seq": entry["sequence"].get("value", ""),
    }
    cross_references = ["PDB", "Pfam", "AlphaFoldDB"]
    data.update(extract_cross_references(entry, cross_references))
    return data


def parse_json_file(json_path, csv_parsed_path):
    if not csv_parsed_path.is_file():
        data = []
        with open(json_path, "r") as file:
            entries = ijson.items(file, "results.item")
            for entry in tqdm(entries, desc="Processing entries"):
                data.append(parse_entry(entry))
        df = pd.DataFrame(data)
        df.to_csv(csv_parsed_path, index=False)
    else:
        df = pd.read_csv(csv_parsed_path)
    return df


def create_fasta_file(df, fasta_path):
    records = df.apply(lambda row: f">{row['uid']}\n{row['seq']}\n", axis=1)
    with open(fasta_path, "w") as handle:
        handle.write("".join(records))


def main():
    parser = argparse.ArgumentParser(
        description="Parse UniProt JSON file and create FASTA and CSV file"
    )
    parser.add_argument("json", help="Input JSON file path")
    args = parser.parse_args()

    input_path = Path(args.json)
    output_csv = input_path.with_suffix(".csv")
    output_fasta = input_path.with_suffix(".fasta")

    df = parse_json_file(input_path, output_csv)
    create_fasta_file(df, output_fasta)

    print(f"Parsed CSV saved to: {output_csv}")
    print(f"FASTA file saved to: {output_fasta}")


if __name__ == "__main__":
    main()
