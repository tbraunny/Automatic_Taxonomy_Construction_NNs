#!/bin/bash

# Check if a directory is passed
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/pdf-directory"
    exit 1
fi

PDF_DIR="$1"

# Check if the path exists and is a directory
if [ ! -d "$PDF_DIR" ]; then
    echo "Error: '$PDF_DIR' is not a directory."
    exit 1
fi

# Loop through all PDFs in the directory
for pdf_file in "$PDF_DIR"/*.pdf; do
    # Check if any pdf exists
    [ -e "$pdf_file" ] || continue

    # Get the filename without path and extension
    filename=$(basename "$pdf_file" .pdf)

    # Create a subdirectory with the same name
    target_dir="$PDF_DIR/$filename"
    mkdir -p "$target_dir"

    # Move the PDF into the new subdirectory
    mv "$pdf_file" "$target_dir/"
done

echo "Done."
