import argparse
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def run_command(command: str) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        print(line, end="", file=sys.stdout)  # Print each line in real-time

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def read_fasta(filename: Path) -> Dict[str, str]:
    sequences = {}
    with open(filename) as file:
        for line in file:
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                sequences[current_id] = []
            else:
                sequences[current_id].append(line.strip())
    return {id_: "".join(seq) for id_, seq in sequences.items()}


def read_clusters(filename: Path) -> Dict[str, List[str]]:
    clusters = defaultdict(list)
    with open(filename) as file:
        for line in file:
            query, target = line.strip().split("\t")
            clusters[query].append(target)
    return dict(clusters)


def create_split_files(
    clusters: Dict[str, List[str]],
    sequences: Dict[str, str],
    output_dir: Path,
    splits: Tuple[float, float, float],
) -> None:
    all_queries = list(clusters.keys())
    random.seed(42)
    random.shuffle(all_queries)

    train_split, val_split = [
        int(split * len(all_queries)) for split in splits[:2]
    ]

    train_queries = set(all_queries[:train_split])
    val_queries = set(all_queries[train_split : train_split + val_split])
    test_queries = set(all_queries[train_split + val_split :])

    for split_name, split_queries in [
        ("train", train_queries),
        ("val", val_queries),
        ("test", test_queries),
    ]:
        with open(output_dir / f"{split_name}_set.fasta", "w") as outfile:
            for query in sorted(split_queries):
                for target in clusters[query]:
                    outfile.write(f">{target}\n{sequences[target]}\n")


def run_mmseqs_easy_cluster(
    fasta_file: Path, cluster_results: Path, tmp_dir: Path, threads: int
):
    print("Running `mmseqs easy-cluster` command...", file=sys.stderr)
    run_command(
        f"mmseqs easy-cluster {fasta_file} {cluster_results} {tmp_dir} -s 7.5 "
        f"--min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 0 --threads {threads}"
    )
    print("`mmseqs easy-cluster` command completed.", file=sys.stderr)


def run_all_against_all_search(
    input_fasta: Path, output_dir: Path, tmp_dir: Path, threads: int
) -> None:
    base_name = input_fasta.stem
    db_name = f"{tmp_dir}/{base_name}_db"
    result_db = f"{tmp_dir}/{base_name}_results"
    search_tmp = f"{tmp_dir}/{base_name}_tmp"
    result_tsv = f"{output_dir}/{base_name}_all_vs_all.tsv"

    print(f"Running all-against-all search for {base_name}...", file=sys.stderr)

    # Create MMseqs2 database
    if not Path(db_name).with_suffix(".dbtype").exists():
        run_command(f"mmseqs createdb {input_fasta} {db_name}")

    # Run all-against-all search
    if not Path(result_db).with_suffix(".dbtype").exists():
        run_command(
            f"mmseqs search {db_name} {db_name} {result_db} {search_tmp} -s 7.5 --threads {threads}"
        )

    # Convert results to human-readable format
    run_command(
        f"mmseqs convertalis {db_name} {db_name} {result_db} {result_tsv} "
        f"--format-mode 4 --format-output query,target,fident,evalue,qcov,tcov"
    )

    print(
        f"All-against-all search completed for {base_name}. Results saved to {result_tsv}",
        file=sys.stderr,
    )


def main(fasta_file: Path, output_dir: Path, threads: int):
    # Create necessary directories
    tmp_dir = output_dir / "tmp"
    cluster_results = output_dir / "cluster" / fasta_file.stem
    cluster_file = cluster_results.with_stem(f"{fasta_file.stem}_cluster.tsv")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cluster_results.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Run MMseqs2 easy-cluster if cluster file doesn't exist
    if not cluster_file.exists():
        run_mmseqs_easy_cluster(fasta_file, cluster_results, tmp_dir, threads)

    # Step 2: Read sequences and clusters
    sequences = read_fasta(fasta_file)
    clusters = read_clusters(cluster_file)

    # Step 3: Create split files (train, validation, test)
    create_split_files(
        clusters, sequences, output_dir, splits=(0.7, 0.15, 0.15)
    )
    print(
        "FASTA files created for train, validation, and test sets.",
        file=sys.stderr,
    )

    # Step 4: Run all-against-all search for each set
    for split in ["train", "val", "test"]:
        split_fasta = output_dir / f"{split}_set.fasta"
        run_all_against_all_search(split_fasta, output_dir, tmp_dir, threads)

    print("All-against-all searches completed for all sets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a FASTA file into train, validation, and test sets using MMseqs2."
    )
    parser.add_argument("fasta_file", type=Path, help="Input FASTA file")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use for MMseqs2 commands (default: 8)",
    )

    args = parser.parse_args()
    main(args.fasta_file, args.output_dir, args.threads)
