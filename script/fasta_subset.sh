#!/bin/bash
# usage: bash fasta_subset.sh input.fasta 5 > subset.fasta

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_fasta_file> <number_of_headers>"
    exit 1
fi

input_file=$1
num_headers=$2

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist."
    exit 1
fi

# Check if the number of headers is a positive integer
if ! [[ "$num_headers" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of headers must be a positive integer."
    exit 1
fi

# Generate the subset
awk -v n="$num_headers" '
    /^>/ {
        header_count++
        if (header_count > n) exit
    }
    { print }
' "$input_file"