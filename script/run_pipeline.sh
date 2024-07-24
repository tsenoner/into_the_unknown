#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <base_directory> <output_directory>"
    exit 1
fi

# Assign input arguments to variables
BASE_DIR=$1
OUTPUT_BASE=$2

# Ensure the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory does not exist."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# Set file paths
INPUT_JSON="$BASE_DIR/$(basename $BASE_DIR).json"
PARSED_CSV="$BASE_DIR/$(basename $BASE_DIR).csv"
FASTA_FILE="$BASE_DIR/$(basename $BASE_DIR).fasta"
FINAL_CSV="$BASE_DIR/$(basename $BASE_DIR)_final.csv"
OUTPUT_DIR="$OUTPUT_BASE/corr_$(basename $BASE_DIR)"

SCRIPT_BASE="into_the_unknown/prepare_data"

# Run the pipeline using Poetry
echo "Starting protein analysis pipeline..."

echo "Step 1: Running 1_parse_file.py..."
poetry run python "$SCRIPT_BASE/1_parse_file.py" "$INPUT_JSON"

echo "Step 2: Running 2_parse_foldseek.py..."
poetry run python "$SCRIPT_BASE/2_parse_foldseek.py" "$BASE_DIR"

echo "Step 3: Running 3_create_plots.py..."
poetry run python "$SCRIPT_BASE/3_create_plots.py" -i "$FINAL_CSV" -o "$OUTPUT_DIR"

echo "Pipeline completed successfully!"
echo "Output files can be found in $OUTPUT_DIR"