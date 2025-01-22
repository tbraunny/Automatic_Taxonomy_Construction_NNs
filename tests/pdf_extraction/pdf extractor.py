import fitz  # PyMuPDF
import re
import json

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    doc = fitz.open(file_path)
    content = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        content.append(page.get_text("text"))
    return "\n".join(content)

def extract_headers_with_format(file_path):
    """Extract headers and their font sizes from the PDF."""
    doc = fitz.open(file_path)
    headers = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    font_size = span["size"]
                    text = span["text"].strip()
                    # Capture potential headers
                    if font_size > 12 and text and len(text) < 100:  # Adjust font size threshold
                        headers.append((font_size, text))

    return headers



def classify_headers(headers):
    """Classify headers into main sections and subsections based on numbering and size."""
    organized_headers = {"Main Sections": [], "Subsections": []}

    for font_size, header in headers:
        if re.match(r"^\\d+(\\.\\d+)*\\s", header):  # Numbered sections
            if "." in header.split()[0]:
                organized_headers["Subsections"].append(header)
            else:
                organized_headers["Main Sections"].append(header)
        elif font_size > 15:  # Capture unnumbered main sections
            organized_headers["Main Sections"].append(header)
        else:
            organized_headers["Subsections"].append(header)

    return organized_headers


def find_figures_and_tables(pdf_text):
    """Find proper references to figures and tables."""
    figures = re.findall(r"Figure\\s+\\d+[:]?.{0,100}", pdf_text)  # Limit to 100 characters
    tables = re.findall(r"Table\\s+\\d+[:]?.{0,100}", pdf_text)   # Limit to 100 characters
    return figures, tables


def save_results(headers, figures, tables, output_file="results.json"):
    """Save extracted results to a JSON file."""
    data = {
        "Headers": headers,
        "Figures": figures,
        "Tables": tables
    }
    print(json.dumps(data, indent=4))  # Debug output to console
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def main():
    # Replace with your PDF file path
    pdf_path = "/home/richw/richie/ATCNN/data/raw/AlexNet.pdf"
    
    # Extract text
    pdf_text = extract_text_from_pdf(pdf_path)

    # Extract headers with font sizes
    headers = extract_headers_with_format(pdf_path)

    # Classify headers into main sections and subsections
    structured_headers = classify_headers(headers)

    # Find figures and tables
    figures, tables = find_figures_and_tables(pdf_text)

    # Save results to a JSON file
    save_results(structured_headers, figures, tables)

    print("Extraction complete. Results saved to 'results.json'.")

if __name__ == "__main__":
    main()
