import csv
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterator, List, Set


def read_fasta_headers(file_path: Path) -> Set[str]:
    with file_path.open("r") as file:
        headers = {line.strip()[1:] for line in file if line.startswith(">")}
    return headers


def read_csv_rows(csv_path: Path) -> Iterator[Dict[str, str]]:
    with csv_path.open("r", newline="") as file:
        yield from csv.DictReader(file)


def process_csv_with_fasta_headers(
    csv_row: Dict[str, str],
    fasta_headers: Set[str],
    required_columns: List[str],
) -> Dict[str, str] | None:
    if csv_row["query"] in fasta_headers and csv_row["target"] in fasta_headers:
        return {col: csv_row[col] for col in required_columns}
    return None


def process_single_fasta(fasta_path: Path, csv_path: Path, output_path: Path):
    print(f"Processing `{fasta_path.name}`")

    required_columns = [
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
        "qlen",
        "tlen",
        "aa_comp_diff",
        "ungapped_len",
        "hfsp",
    ]

    fasta_headers = read_fasta_headers(fasta_path)
    print(f"Found {len(fasta_headers):,} headers in `{fasta_path.name}`")

    processed_count = 0
    with output_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=required_columns)
        writer.writeheader()

        for row in read_csv_rows(csv_path):
            if processed_row := process_csv_with_fasta_headers(
                row, fasta_headers, required_columns
            ):
                writer.writerow(processed_row)
                processed_count += 1

    print(
        f"Processed {processed_count:,} rows for `{fasta_path.name}` saved to `{output_path}`"
    )


def process_all_files(
    base_dir: Path, foldseek_csv_file: Path, fasta_files: Dict[str, Path]
):
    output_dir = base_dir / "training"
    output_dir.mkdir(exist_ok=True)

    tasks = [
        (fasta_path, foldseek_csv_file, output_dir / f"{name}.csv")
        for name, fasta_path in fasta_files.items()
    ]

    print(f"Starting processing of {len(tasks)} FASTA files")
    with Pool(cpu_count()) as pool:
        pool.starmap(process_single_fasta, tasks)
    print(f"All processing completed.")


if __name__ == "__main__":
    base_dir = Path("data/swissprot")
    foldseek_csv_file = base_dir / "swissprot_final.csv"
    fasta_files = {
        "train": base_dir / "mmseqs" / "train_set.fasta",
        "val": base_dir / "mmseqs" / "val_set.fasta",
        "test": base_dir / "mmseqs" / "test_set.fasta",
    }

    process_all_files(base_dir, foldseek_csv_file, fasta_files)
