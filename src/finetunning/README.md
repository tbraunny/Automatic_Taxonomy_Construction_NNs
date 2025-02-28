# Instructions to finetune
1. Follow instructions to install: https://github.com/huggingface/autotrain-advanced
2. Create a directory called dataset and data
3. Put all your data in the data directory and run chunking.py -- a chunking.csv will be in data
4. Now finetune ```bash CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=torch autotrain --config qwenl.yml``` 
5. Get some coffee and donut and wait assuming the train process works
6. Now convert the model into a gguf file with ggufcreation.py
7. Alter modelfile to point to the created gguf file that you made with ggufcreation.py -- preferably the 4bit quantization one
8. Now create the ollama model with ```bash ollama create -f modelfile neuralexpert  ```
