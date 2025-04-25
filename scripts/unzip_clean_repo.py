import zipfile
import os
import shutil

def unzip_and_clean_repo(zip_path, extract_to):
    """
    Extracts a GitHub repository ZIP file and removes all .git directories
    and .gitignore files recursively.

    Args:
        zip_path (str): Path to the ZIP file.
        extract_to (str): Directory to extract the contents to.
            If None, extracts to a directory with the same name as the ZIP file (minus extension).
    """
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Recursively remove .git and .gitignore
    for root, dirs, files in os.walk(extract_to, topdown=False):
        # Remove .git directories
        if '.git' in dirs:
            shutil.rmtree(os.path.join(root, '.git'))
        # Remove .gitignore files
        if '.gitignore' in files:
            os.remove(os.path.join(root, '.gitignore'))
        if '.gitattributes' in files:
            os.remove(os.path.join(root , '.gitattributes'))

    print(f"Extraction and cleanup completed in '{extract_to}'")

