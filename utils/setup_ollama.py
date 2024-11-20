# Install Ollama in WSL: curl -fsSL https://ollama.com/install.sh | sh

# Note: allocate max CPU cores and RAM to WSL

import os


"""
Utility script to start and stop ollama server
"""

'''
Example usage:
from utils.setup_ollama import OllamaServer

start_ollama_server()
check_and_load_model(model_name='llama3.1:8b'):
stop_ollama_server()
'''


class OllamaServer:
    def __init__(self,model_name='llama3.2:1b'):
        self.model_name=model_name

    def start_ollama_server(self):
        print("Starting Ollama server...")
        import subprocess
        result = subprocess.run("ollama serve", shell=True)
        if result.returncode == 0:
            print("Ollama server started successfully.")
        else:
            print("Failed to start Ollama server.")

    def stop_ollama_server(self):
        """
        Stops the Ollama server.
        :return: None
        """
        print(f"Stopping Ollama server for model {self.model_name}...")
        os.system(f"ollama stop {self.model_name}")

    def check_and_load_model(self):
        """
        Checks if the specified Ollama model is loaded. If not, it pulls the model.
        :param model_name: Name of the Ollama model to check or load
        :type model_name: string
        :return: None
        """
        os.system(f"ollama pull {self.model_name}")
