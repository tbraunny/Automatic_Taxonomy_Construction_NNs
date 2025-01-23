import os
from utils.file_utils import read_file, write_json_file
from utils.markdown_utils import parse_markdown_to_structure, remove_excluded_sections

# Keyword headers to exclude
EXCLUDED_SECTIONS = [
    "References", "Citations", "Related Works", "Authors", "Background"
]

def process_file(file_path, output_path):
    """
    Processes a markdown file and writes the filtered structure as JSON.
    :param file_path: Path to the markdown file.
    :param output_path: Path to the output JSON file.
    """
    markdown_content = read_file(file_path)
    structured_data = parse_markdown_to_structure(markdown_content)
    filtered_data = remove_excluded_sections(structured_data, EXCLUDED_SECTIONS)
    write_json_file(filtered_data, output_path)

if __name__ == "__main__":
    # Input and output file paths
    input_file = '/home/richw/richie/ATCNN/tests/pdf_extraction/alexnet.md'
    output_file = '/home/richw/richie/ATCNN/data/parsed_json/Parsed_AlexNet.json'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process the file
    process_file(input_file, output_file)

    print(f"Parsed and filtered data has been written to {output_file}")
