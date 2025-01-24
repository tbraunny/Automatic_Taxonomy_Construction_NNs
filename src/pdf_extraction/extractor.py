import os
from utils.file_utils import read_file, write_json_file
from utils.txt_utils import parse_text_to_header_content, remove_excluded_sections
from utils.docling_pdf_loader import DoclingPDFLoader

"""
Point of entry code to extract headers associated sections into json format
as well as remove unneeded sections such as references and related works

To use:
1. Ensure that a pdf format of a paper is in data/raw/
2. Define the 'ann_name' variable in main to the file name of pdf
3. Run code

Output: 
Extracted pdf outputed to data/extracted_markdown/
Parsed header and sections saved to data/parsed_json/
"""


## Addresses TORCH_CUDA_ARCH_LIST warning (not necessary; for optimization)
# export TORCH_CUDA_ARCH_LIST="8.6" ## For 3090 or 4070
# Currently added to ~/.bashrc for persitance


EXCLUDED_SECTIONS = [
    "References", "Citations", "Related Works", "Authors", "Background"
]

def extract_pdf_to_markdown(pdf_path, md_output_path):
    """Extracts text from a PDF and writes it to a Markdown file"""
    loader = DoclingPDFLoader(file_path=pdf_path)


    # Load  documents
    docs = loader.load()

    # Save doc text to markdown
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

    # Must change this name to the paper pdf name
    ann_name = 'resnet'
    PDF_PATH = f"/home/richw/richie/ATCNN/data/{ann_name}/{ann_name}.pdf"
    MD_PATH = f"data/{ann_name}/markdown_{ann_name}.md"
    JSON_PATH = f"data/{ann_name}/parsed_{ann_name}.json"

    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

    extract_pdf_to_markdown(PDF_PATH, MD_PATH)
    process_markdown_to_json(MD_PATH, JSON_PATH)
