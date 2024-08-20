#!/bin/bash
set -e

# Check if the foldcomp file path is provided as an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <foldcomp_file_path>"
  exit 1
fi

# --- variables ---
FOLDCOMP_FILE="$1"
TMP_DIR="/scratch/tobias/foldseek"
DATABASE_NAME="$TMP_DIR/foldseek_db"
RESULT_DB="$TMP_DIR/result"
SEARCH_TMP="$TMP_DIR/search_tmp"
RESULT_TSV=$(echo $FOLDCOMP_FILE | sed 's|[^/]*$|result.tsv|')

# --- parameters ---
INPUT_FORMAT=5 # 5 = Foldcomp format
THREADS=128
COV_MODE=0
SENSITIVITY=7.5

# Create tmp directory if it doesn't exist
mkdir -p $TMP_DIR

# Step 1: Create Foldseek database
foldseek createdb $FOLDCOMP_FILE $DATABASE_NAME --input-format $INPUT_FORMAT \
  --threads $THREADS

# Step 2: Run all-against-all search
foldseek search $DATABASE_NAME $DATABASE_NAME $RESULT_DB $SEARCH_TMP \
  --cov-mode $COV_MODE -a 1 --threads $THREADS -s $SENSITIVITY \
  --exhaustive-search 1 --max-seqs 10000 --remove-tmp-files 1

# Step 3: Save reults as TSV file
foldseek convertalis $DATABASE_NAME $DATABASE_NAME $RESULT_DB $RESULT_TSV \
  --format-mode 4 --threads $THREADS --format-output \
  query,target,fident,nident,alnlen,mismatch,evalue,qcov,tcov,lddt,rmsd,alntmscore,qtmscore,ttmscore

# qstart,qend,tstart,tend,bits,cigar,qcov,tcov
#--db-output 0 --compressed 0 --exact-tmscore 1

# Step 4: Copy results to local disk
cp $RESULT_TMP $RESULT_TSV

echo "All-against-all search completed. Results are stored in $OUTPUT_FOLDER."
