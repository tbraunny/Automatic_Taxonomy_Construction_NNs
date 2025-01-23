import re
import json

def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Keyword headers to remove
excluded_sections = [
    "References", "Citations", "Related Works", "Authors", "Background"
]

# Parse text into dictionary
def parse_markdown_to_structure(markdown):
    structured_data = [] 
    current_section = {
        "header": None, 
        "content": []   
    }

    # Iterate through each line of the text
    for line in markdown.splitlines():
        # Check if the line is a header (e.g., "## Section Title")
        header_match = re.match(r"^(##+)\s+(.*)$", line)

        if header_match:
            # If a header is found, save the current section (if it exists)
            if current_section["header"]:
                structured_data.append(current_section)

            # Start a new section with the current header
            current_section = {
                "header": header_match.group(2).strip(),
                "content": []
            }
        else:
            # Append non-header lines to the current section's content
            current_section["content"].append(line)

    # Add the last section to the structured data if it has a header
    if current_section["header"]:
        structured_data.append(current_section)

    return structured_data

# Filters out sections with headers matching excluded keywords
def remove_excluded_sections(structured_data, excluded_keywords):
    return [
        section for section in structured_data
        if not any(excluded.lower() in section["header"].lower() for excluded in excluded_keywords)
    ]

# Processes a file and returns the filtered sections
def process_file(file_path):
    markdown_content = read_markdown_file(file_path)
    
    structured_data = parse_markdown_to_structure(markdown_content)
    
    filtered_data = remove_excluded_sections(structured_data, excluded_sections)
    
    return filtered_data

file_path = '/home/richw/richie/ATCNN/tests/pdf_extraction/alexnet.md' 
parsed_structure = process_file(file_path)

# Output the parsed and filtered structure as JSON
with open('/home/richw/richie/ATCNN/data/parsed_json/Parsed_AlexNet.json', 'w') as json_file:
    json.dump(parsed_structure, json_file, indent=4)
