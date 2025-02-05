import torch

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available. Using CPU.")

import langchain
print(langchain.__version__)
