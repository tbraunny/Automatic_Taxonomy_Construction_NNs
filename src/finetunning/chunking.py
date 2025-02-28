'''
Simple program to parse 
'''
import csv
import glob
import os
from docling.document_converter import DocumentConverter

def split_text(text,chunk=2048):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def parse_pdf(path):
    converter = DocumentConverter()
    result = converter.convert(source)
    output = result.document.export_to_markdown()
    return output

if __name__ == '__main__':

    data = glob.glob('data/*.txt') + glob.glob('data/*.md') + glob.glob('data/*.pdf')
    chunked_text = []
    for path in data:
        if path.endwith('.pdf'):
            text = parse_pdf(path)
        else:
            text = open(path,'r').read()
        chunked_text += split_text(text)

    # Write each chunk to a CSV file, one chunk per row
    output_csv = "dataset/chunks.csv"

    with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text"])  # CSV header
        
        for chunk in chunks:
            writer.writerow([chunk])

    print(f"Created '{output_csv}' with {len(chunks)} rows.")
