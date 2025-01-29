import pickle
import os

def save_to_pickle(obj, file_path: str):
    """Saves an object to a pickle file."""
    assert file_path.endswith('.pkl'), "File path must have a .pkl extension"
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {file_path}")

def load_from_pickle(file_path: str):
    """Loads an object from a pickle file."""
    assert file_path.endswith('.pkl'), "File path must have a .pkl extension"
    assert os.path.isfile(file_path), "File does not exist"
    
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {file_path}")
    return obj
