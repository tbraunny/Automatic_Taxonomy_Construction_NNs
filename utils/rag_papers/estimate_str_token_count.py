from transformers import AutoTokenizer

def estimate_str_tokens_hf(text: str, model_name: str = "meta-llama/Llama-3.1-8B"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_count = tokenizer.encode(text)
        print(f"Estimated token count: {token_count}")

    except Exception as e:
        print(f"Error estimating tokens with Hugging Face tokenizer: {e}")
        return None
