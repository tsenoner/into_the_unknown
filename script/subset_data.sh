#!/bin/bash

subset_csv() {
    local input_file="$1"
    local output_file="$2"
    local subset_size="${3:-100}"  # in percentage

    # Check if required arguments are provided
    if [[ -z "$input_file" || -z "$output_file" ]]; then
        echo "Usage: subset_csv <input_file> <output_file> [subset_size_percentage]"
        return 1
    fi

    # Ensure the input file exists and is readable
    if [[ ! -f "$input_file" || ! -r "$input_file" ]]; then
        echo "Error: Input file '$input_file' not found or not readable!"
        return 1
    fi

    # Validate subset size
    if ! [[ "$subset_size" =~ ^[0-9]+([.][0-9]+)?$ ]] || (( $(echo "$subset_size < 0.01 || $subset_size > 100" | bc -l) )); then
        echo "Error: Subset size must be a number between 0.01 and 100."
        return 1
    fi

    # Check if output file already exists
    if [[ -f "$output_file" ]]; then
        read -p "Output file '$output_file' already exists. Overwrite? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Operation cancelled."
            return 1
        fi
    fi

    # Get the header from the input file and write to the output file
    head -n 1 "$input_file" > "$output_file"

    # Calculate the number of lines to sample
    local total_lines
    total_lines=$(wc -l < "$input_file")
    local sample_lines
    sample_lines=$(echo "($total_lines - 1) * $subset_size / 100" | bc -l)
    sample_lines=$(printf "%.0f" "$sample_lines")  # Convert to integer

    # Ensure at least one line is sampled if the calculation rounds to 0
    if [[ $sample_lines -eq 0 ]]; then
        sample_lines=1
    fi

    # Shuffle and sample the required number of lines from the input file, excluding the header
    tail -n +2 "$input_file" | shuf -n "$sample_lines" >> "$output_file"

    echo "Subset created successfully: $sample_lines lines (${subset_size}%) written to $output_file"
}

# Call the function with the arguments passed to the script
subset_csv "$@"
