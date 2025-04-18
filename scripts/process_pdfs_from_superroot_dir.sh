#!/bin/bash

# Check if base directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <base_directory>"
  exit 1
fi

BASE_DIR="$1"

# Check if the provided argument is a directory
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: $BASE_DIR is not a directory."
  exit 1
fi

# Iterate over each subdirectory
for SUBDIR in "$BASE_DIR"/*; do
  if [ -d "$SUBDIR" ]; then
    NAME=$(basename "$SUBDIR")
    PDF_PATH="$SUBDIR/$NAME.pdf"

    # Check if the PDF exists
    if [ -f "$PDF_PATH" ]; then
      echo "Processing $PDF_PATH"
      python3 src/pdf_extraction/extract_filter_pdf_to_json.py --pdf_path "$PDF_PATH"
    else
      echo "Warning: PDF not found for $NAME in $SUBDIR"
    fi
  fi
done
