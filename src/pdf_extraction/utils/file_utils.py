import json

def read_file(file_path, encoding='utf-8'):
    """
    Reads the content of a file.
    :param file_path: Path to the file to be read.
    :param encoding: Encoding of the file.
    :return: Content of the file as a string.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()

def write_json_file(data, file_path, indent=4):
    """
    Writes data to a JSON file.
    :param data: Data to write (typically a dictionary or list).
    :param file_path: Path to the JSON file to write to.
    :param indent: Indentation level for pretty printing.
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=indent)
