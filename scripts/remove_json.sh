#!/bin/bash

# Check if a directory was provided
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/root_directory"
  exit 1
fi

ROOT_DIR="$1"

# Verify that the provided path is a directory
if [ ! -d "$ROOT_DIR" ]; then
  echo "Error: '$ROOT_DIR' is not a directory."
  exit 1
fi

# Find and delete all .json files recursively
find "$ROOT_DIR" -type f -name "*.json" -exec rm -f {} +

echo "All .json files have been removed from '$ROOT_DIR'."
