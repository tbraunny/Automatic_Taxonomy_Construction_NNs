import os
import ast
import shutil
from pathlib import Path

def resolve_import_path(current_file, import_node, root_dir):
    """
    Resolve an ast.Import or ast.ImportFrom into file paths under root_dir.
    Returns a list of candidate .py files or package __init__.py files.
    """
    root_dir    = Path(root_dir).resolve()
    current_dir = Path(current_file).parent.resolve()
    candidates  = []

    # --- Handle `from ... import ...` ---
    if isinstance(import_node, ast.ImportFrom):
        # only ImportFrom has .level and .module
        level  = import_node.level     # 0=absolute, 1=current pkg, 2=one up, etc.
        module = import_node.module    # None for "from . import foo"

        # 1) pick base directory
        if level == 0:
            # absolute import: start at repo root
            base = root_dir
        else:
            # relative import: start in file's folder, climb (level-1) times
            base = current_dir
            for _ in range(level - 1):
                base = base.parent

        # 2a) from X import ...
        if module:
            parts   = module.split('.')
            modpath = base.joinpath(*parts)

            # module file?
            py = modpath.with_suffix('.py')
            if py.exists():
                candidates.append(str(py))
            # or package __init__.py?
            init = modpath / '__init__.py'
            if init.exists():
                candidates.append(str(init))

        # 2b) from . import a, b, c
        else:
            for alias in import_node.names:
                apath = base / alias.name
                py    = apath.with_suffix('.py')
                if py.exists():
                    candidates.append(str(py))
                init  = apath / '__init__.py'
                if init.exists():
                    candidates.append(str(init))

    # --- Handle `import foo.bar` ---
    elif isinstance(import_node, ast.Import):
        # absolute import—always start at repo root
        for alias in import_node.names:
            parts   = alias.name.split('.')
            modpath = root_dir.joinpath(*parts)

            py   = modpath.with_suffix('.py')
            if py.exists():
                candidates.append(str(py))
            init = modpath / '__init__.py'
            if init.exists():
                candidates.append(str(init))

    return candidates


def find_dependencies(file_path, root_dir):
    """
    Recursively find all local Python files imported by `file_path`.
    """
    visited = set()
    to_process = [str(Path(file_path).resolve())]

    while to_process:
        current = to_process.pop()
        if current in visited:
            continue
        visited.add(current)

        try:
            with open(current, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=current)
        except (SyntaxError, FileNotFoundError):
            continue  # Skip invalid files

        for node in ast.walk(tree):
            if isinstance(node, (ast.ImportFrom, ast.Import)):
                paths = resolve_import_path(current, node, root_dir)
                for p in paths:
                    if p.startswith(str(Path(root_dir).resolve())):
                        to_process.append(p)

    return visited

def extract_dependencies(root_dir, target_filename, output_dir):
    """
    Copy only the Python files needed by `target_filename` from `root_dir`
    into `output_dir`, preserving the directory structure.
    """
    root_dir = Path(root_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # 1) locate the target file in root_dir
    target_path = None
    for p in root_dir.rglob(target_filename):
        if p.is_file():
            target_path = p
            break
    if target_path is None:
        raise FileNotFoundError(f"Could not find {target_filename} under {root_dir}")

    # 2) compute the full set of files to copy
    deps = find_dependencies(str(target_path), str(root_dir))
    deps.add(str(target_path))

    # 3) for each file, re-create its relative path under output_dir
    for abs_path in deps:
        src = Path(abs_path)
        # compute relative path w.r.t. the repo root
        rel = src.relative_to(root_dir)
        dst = output_dir / rel

        # make sure the destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        # copy the file (preserve metadata)
        shutil.copy2(src, dst)

    print(f"Copied {len(deps)} files into {output_dir}")


def clean_directory(root_dir, target_filename):
    """
    Keeps only the necessary files for the target file to work (based on import dependencies).
    """
    root_dir = str(Path(root_dir).resolve())
    target_file = None

    # Find target file by filename
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name == target_filename:
                target_file = os.path.join(dirpath, name)
                break
        if target_file:
            break

    if not target_file:
        print(f"Target file {target_filename} not found in {root_dir}")
        return

    # Get all dependencies
    keep_files = find_dependencies(target_file, root_dir)
    keep_files.add(str(Path(target_file).resolve()))

    # Delete everything else
    for dirpath, _, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if str(Path(full_path).resolve()) not in keep_files:
                os.remove(full_path)
        # Optionally remove empty dirs
        if not os.listdir(dirpath):
            os.rmdir(dirpath)

    print(f"Kept {len(keep_files)} files related to {target_filename}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean repo to keep only needed files for a target PyTorch nn.Module file.")
    parser.add_argument("root_dir", help="Path to the root of the repository")
    parser.add_argument("target_filename", help="The Python file containing the nn.Module you want to trace")


    args = parser.parse_args()
    extract_dependencies(args.root_dir, args.target_filename, "ann_name")

    # base.py
    # PyTorch-VAE