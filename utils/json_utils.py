import json
from typing import Any

def get_json_value(key: str, filename: str) -> Any:
    """
    Reads a JSON file and returns the value of the given key.

    :param key: The key.
    :param filename: The filename where the JSON data is stored.
    :return: The value if found, otherwise "Key not found".
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get(key, "Key not found")
    except FileNotFoundError:
        return "File not found"
    except json.JSONDecodeError:
        return "Invalid JSON file"