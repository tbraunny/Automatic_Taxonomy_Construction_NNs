import re

def parse_text_to_header_content(text):
    """
    Parses text content into a structured dictionary format.
    :param str: Raw text file content as a string.
    :return: List of dictionaries representing sections.
    """
    structured_data = [] 
    current_section = {"header": None, "content": []}

    for line in text.splitlines():
        header_match = re.match(r"^(##+)\s+(.*)$", line)
        if header_match:
            if current_section["header"]:
                structured_data.append(current_section)
            current_section = {
                "header": header_match.group(2).strip(),
                "content": []
            }
        else:
            current_section["content"].append(line)

    if current_section["header"]:
        structured_data.append(current_section)

    return structured_data

def remove_excluded_sections(structured_data, excluded_keywords):
    """
    Filters out sections whose headers match excluded keywords.
    :param structured_data: List of dictionaries representing sections.
    :param excluded_keywords: List of keywords to exclude.
    :return: Filtered structured data.
    """
    return [
        section for section in structured_data
        if not any(excluded.lower() in section["header"].lower() for excluded in excluded_keywords)
    ]
