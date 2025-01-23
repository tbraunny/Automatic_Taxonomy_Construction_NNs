import os
from utils.file_utils import read_file, write_json_file
from utils.txt_utils import parse_text_to_header_content, remove_excluded_sections
from docling_pdf_loader import DoclingPDFLoader

EXCLUDED_SECTIONS = [
    "References", "Citations", "Related Works", "Authors", "Background"
]

def extract_pdf_to_markdown(pdf_path, md_output_path):
    """Extracts text from a PDF and writes it to a Markdown file"""
    loader = DoclingPDFLoader(file_path=pdf_path)


    # Load and split documents
    docs = loader.load()

    # Save split text to Markdown
    with open(md_output_path, "w") as f:
        for doc in docs:
            f.write(doc.page_content)
            f.write("\n---\n")
    print(f"Extracted pdf written to {md_output_path} as markdown")


def process_markdown_to_json(md_path, json_output_path):
    """Parses a Markdown file and writes filtered content as JSON"""
    markdown_content = read_file(md_path)
    structured_data = parse_text_to_header_content(markdown_content)
    filtered_data = remove_excluded_sections(structured_data, EXCLUDED_SECTIONS)
    write_json_file(filtered_data, json_output_path)
    print(f"Parsed and filtered data written to {json_output_path}")


if __name__ == "__main__":
    PDF_PATH = "/home/richw/richie/ATCNN/data/raw/AlexNet.pdf"
    MD_PATH = "/home/richw/richie/ATCNN/data/extracted_markdown/alexnet.md"
    JSON_PATH = "/home/richw/richie/ATCNN/data/parsed_json/parsed_alexnet.json"

    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

    extract_pdf_to_markdown(PDF_PATH, MD_PATH)
    process_markdown_to_json(MD_PATH, JSON_PATH)
