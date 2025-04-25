import os

def join_unique_dir(dir: str, dir2: str):
    """
    """
    os.makedirs(dir,exist_ok=True)

    path = os.path.join(dir, dir2)


    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{path}_{i}"):
            i+=1
        path = f"{path}_{i}"
    
    os.makedirs(path)
    return path