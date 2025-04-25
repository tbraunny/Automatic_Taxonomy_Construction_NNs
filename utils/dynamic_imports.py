import importlib
import subprocess
import sys

def install_imports(package_names):
    for pkg in package_names:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"Installing missing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])