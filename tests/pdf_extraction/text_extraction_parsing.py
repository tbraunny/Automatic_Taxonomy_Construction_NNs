import fitz
import pytesseract 
from PIL import Image
import re
import os

def extract_text_from_pdf(pdf_path, output_dir="output_images", ocr_lang="eng"):
    """
    Extracts text from a PDF file robustly, handling both text-based and image-based PDFs.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to store images (for OCR).
        ocr_lang (str): Language for OCR (default: English).
    
    Returns:
        str: The extracted text from the PDF.
    """
    def ocr_image(image):
        """Perform OCR on a PIL Image."""
        return pytesseract.image_to_string(image, lang=ocr_lang)

    # Create output directory for images if OCR is needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize variables
    extracted_text = ""
    pdf_document = fitz.open(pdf_path)
    
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        
        # Try extracting text directly
        text = page.get_text()
        if text.strip():  # If direct text extraction works
            extracted_text += text + "\n"
        else:
            # Fallback to OCR for image-based content
            print(f"Performing OCR on page {page_number + 1}...")
            # Save page as image
            image_path = os.path.join(output_dir, f"page_{page_number + 1}.png")
            pix = page.get_pixmap()
            pix.save(image_path)

            # Open image and perform OCR
            with Image.open(image_path) as img:
                extracted_text += ocr_image(img) + "\n"

    pdf_document.close()
    return extracted_text


def process_pdf(pdf_path):
    """
    High-level function to process a PDF file and extract structured text.

    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        list: A list of cleaned text blocks.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    
    text_blocks = preprocess_research_paper(raw_text)
    
    return text_blocks

def preprocess_research_paper(text):
    """
    Text preprocessing for research papers.
    Handles authors, references, citations, section headers, and common artifacts.

    Args:
        text (str): Raw text extracted from a PDF.

    Returns:
        list: A list of cleaned, coherent text chunks.
    """

    # Lowercase Transformation
    text = text.lower()

    # Remove Special Characters
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # # Remove references, citations, and common noise
    # text = re.sub(r'\[\d+\]', '', text)  # Inline citations like [1], [2]
    # text = re.sub(r'\((.*?)\d{4}(.*?)\)', '', text)  # Inline citations like (Author, 2020)
    # text = re.sub(r'(Figure\s*\d+[:]?.*|Table\s*\d+[:]?.*)', '', text)  # Figures and tables
    # text = re.sub(r'References\n.*', '', text, flags=re.DOTALL)  # References section

    # Remove common metadata patterns (authors, emails, affiliations)
    # text = re.sub(r"\b[A-Za-z\s]+(University|Institute|Lab|Department|Center|School|College)[^\n]*", "", text)

    # Remove references and bibliography sections
    text = re.sub(r"(References|Bibliography|Acknowledgments).*$", "", text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)  # Emails
    text = re.sub(r"(https?://[^\s]+|www\.[^\s]+)", "", text)  # URLs

    # Remove inline citations (e.g., [1], [2], (Author, 2020))
    text = re.sub(r"\[\d+(,\s*\d+)*\]", "", text)  # Inline numbered citations
    text = re.sub(r"\(([^()]*\d{4}[^()]*)\)", "", text)  # Parenthetical citations with author and year

    # Remove section headers and numbering
    text = re.sub(r"^\d+(\.\d+)*\s+[^\n]+", "", text, flags=re.MULTILINE)

    # Remove mathematical expressions
    text = re.sub(r"\$.*?\$", "", text)  # Inline math
    text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", "", text, flags=re.DOTALL)
    text = re.sub(r"[a-zA-Z_]+\s*=\s*[a-zA-Z0-9_+*/^.-]+", "", text)  # Simple math expressions

    # Remove references and citations
    text = re.sub(r"\[\d+\]", "", text)  # Inline citations
    text = re.sub(r"\((.*?)\d{4}(.*?)\)", "", text)  # Parenthetical citations



    # Replace split words (e.g., "Luck- ily" -> "Luckily")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Normalize ligatures and special characters
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "’": "'", "“": '"', "”": '"'}
    for ligature, replacement in ligatures.items():
        text = text.replace(ligature, replacement)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split text into coherent chunks (e.g., paragraphs or sentences)
    chunks = [chunk.strip() for chunk in text.split(". ") if chunk.strip()]

    return chunks

def chunks_to_str(chunks):
    structured_text_string = ""
    for _, block in enumerate(chunks, 1):
        structured_text_string += f"{block}\n"
    return structured_text_string



if __name__ == "__main__":
    pdf_file = "/home/richw/richie/ATCNN/data/raw/ResNet.pdf"
    output_file = "processed_text.txt"
    
    # Process the PDF and extract structured text
    processed_text_blocks = process_pdf(pdf_file)
    processed_text = chunks_to_str(processed_text_blocks)

    # Write the string to the file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(processed_text)