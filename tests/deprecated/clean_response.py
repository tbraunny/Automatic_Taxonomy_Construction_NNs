import json
import re

def clean_response(response, expected_type="atomic"):
    """
    Cleans and parses the response based on the expected type (atomic or property list).

    Args:
        response (str): The raw response text from the model.
        expected_type (str): The expected type of the response ('atomic' or 'property_list').

    Returns:
        str or list: Cleaned atomic value as a string, or a list of properties if expected type is 'property_list'.
    """
    if expected_type == "atomic":
        # For atomic values, return the response as a stripped string
        return response.strip()

    elif expected_type == "property_list":
        try:
            # Attempt to parse as JSON for property lists
            cleaned_response = json.loads(response)
            if isinstance(cleaned_response, list):
                return cleaned_response
            elif isinstance(cleaned_response, dict):
                # If JSON is a dictionary, return only the keys as the property list
                return list(cleaned_response.keys())
        except json.JSONDecodeError:
            # Fallback to regex-based extraction if JSON parsing fails
            # Extract property names assuming a format like `{ 'hasNetwork', 'numLayers' }`
            properties = re.findall(r"'(\w+)'", response)  # Matches property names enclosed in quotes
            return properties

    # Fallback to return response as-is if no expected type matched
    return response

# Example Usage
# Assuming the model response is something like:
# For atomic: "AlexNet"
# For complex: '{"hasNetwork": "class", "hasParameter": "atomic"}'

atomic_response = "AlexNet"
complex_response = '{"hasNetwork": "class", "hasParameter": "atomic"}'

print(clean_response(atomic_response, expected_type="atomic"))  # Output: "AlexNet"
print(clean_response(complex_response, expected_type="complex"))  # Output: {"hasNetwork": "class", "hasParameter": "atomic"}
