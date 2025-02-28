from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("./autotrain-deepseek")
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method = "q4_k_m")
