from llmsherpa.readers import LayoutPDFReader
from rapidfuzz import fuzz
import os

"""
Script for processing PDF files using llama-index's PDFReader docker container.

## Steps to Start the llama-index PDFReader Docker Container

### Pull the Docker Container
    docker pull ghcr.io/nlmatics/nlm-ingestor:latest

### Run the Docker Container
    docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest

### Stop and Remove Docker Containers
    # Stop all running containers
    docker stop $(docker ps -a -q)
    
    # Remove all stopped containers
    docker rm $(docker ps -a -q)

This script processes PDF files, omitting semantic sections like "References" or "Related Works," and writes the cleaned content to a file.
"""


def preprocess_pdf(pdf_path: str, to_file: bool = False):
    """
    Preprocesses a PDF file, parsing and filtering its content.

    Args:
        pdf_path (str): Path to the PDF file.
        to_file (bool): Condition to write parsed PDF to data/preprocessed/{pdf_name}.txt

    Returns:
        Document: The preprocessed document object.
    """
        # Define the llama-index API URL with OCR and new indent parser
    llmsherpa_api_url = (
        "http://localhost:5010/api/parseDocument?"
        "renderFormat=all&useNewIndentParser=yes"
    )
    # Initialize the PDF reader
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)

    # Parse the PDF file
    doc = pdf_reader.read_pdf(pdf_path)

    # Omit unwanted sections and modify the document directly
    omit_semantic_sections(doc)

    if to_file:
        write_to_file(doc, pdf_path)

    # Return the modified document object
    return doc


""" Helper functions """


def omit_semantic_sections(doc):
    """
    Modifies the document object by removing sections that semantically match unwanted keywords,
    and removes tables and figures from the sections.

    Args:
        doc (Document): The parsed document object.
    """
    # Keywords for unwanted sections
    unwanted_keywords = [
        "references",
        "related works",
        "citations",
        "prior work",
        "background",
        "literature review",
        "state of the art",
        "acknowledgments",
    ]

    # Filter out unwanted sections
    filtered_sections = []

    # Iterate over each section in the document
    for section in doc.sections():
        title = section.title.lower() if section.title else ""  # Normalize title to lowercase

        # Check for semantic similarity with unwanted keywords
        if any(fuzz.partial_ratio(title, keyword) > 80 for keyword in unwanted_keywords):
            print(f"Skipping section: {section.title}")
            continue  # Skip sections matching unwanted keywords

        # Filter out tables and figures from the section's children
        filtered_children = []
        for child in section.children:
            if hasattr(child, "type") and child.type in ["Table", "Figure"]:
                print(f"Skipping {child.type} in section {section.title}")
                continue  # Skip tables and figures
            else:
                filtered_children.append(child)

        # Update the section's children
        section.children = filtered_children

        # Append the section to the filtered_sections list
        filtered_sections.append(section)

    # Update the document's sections
    doc.sections_list = filtered_sections  # Assuming doc.sections_list is the correct attribute

    # No return statement needed since we're modifying the doc object directly


def write_to_file(doc, input_path, output_dir="data/preprocessed"):
    """
    Writes text content of the document to a file in the specified output directory.

    Args:
        doc (Document): The document object to write.
        input_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the output file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the original file name without extension
    file_name = os.path.splitext(os.path.basename(input_path))[0]

    # Define the output file path
    output_path = os.path.join(output_dir, f"{file_name}.txt")

    # Collect text from the document
    text_content = []
    for section in doc.sections():
        if section.title:
            text_content.append(f"## {section.title}")
        for child in section.children:
            if hasattr(child, "to_text"):
                text_content.append(child.to_text())

    # Join all collected text with double newlines
    text = "\n\n".join(text_content)

    # Write the content to the file
    with open(output_path, "w") as file:
        file.write(doc.to_text())

    print(f"Preprocessed content written to '{output_path}'")


if __name__ == "__main__":
    # Path to the input PDF file
    input_pdf_path = "data/raw/AlexNet.pdf"

    processed_doc = preprocess_pdf(input_pdf_path)
