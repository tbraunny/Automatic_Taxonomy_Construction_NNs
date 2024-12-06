import json
import re

def extract_JSON(response: str) -> dict:
    """
    Extracts JSON data from a response string.

    Args:
        response (str): The LLM's response containing JSON data.

    Returns:
        dict: Extracted JSON object.

    Raises:
        ValueError: If no valid JSON block is found in the response.
    """
    try:
        json_match = re.search(r'```json\n({.*?})\n```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        raise ValueError("No valid JSON block found in the response.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}\nResponse: {response}")
    
"""
json_dict = extract_JSON(string_to_parse)
value = json_dict.key

"""