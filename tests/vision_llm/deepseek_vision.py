import torch
import os
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# 🧹 Clear GPU memory before execution
torch.cuda.empty_cache()

# ✅ Check GPU info
print("Checking GPU status...")
os.system("nvidia-smi")

# ✅ Specify the model path
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"

# ✅ Load processor
print("Loading processor...")
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# ✅ Load model with memory-efficient settings
print("Loading model...")
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float16,  # More memory efficient
    device_map="auto"           # Auto-allocate across GPUs
).eval()

# ✅ Define conversation prompt
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>List each layer of the residual network sequentially. Do not generalize internal layers, and include all types of layers such as modification and activation layers.",
        "images": ["tests/vision_llm/Resnet Layers.png"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]

# ✅ Load images
print("Loading images...")
pil_images = load_pil_images(conversation)

# ✅ Prepare input data
print("Preparing inputs...")
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=False  # Set to False to save memory
)

# ✅ Move only tensor values to the correct device
device = vl_gpt.device
prepare_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prepare_inputs.items()}

# ✅ Generate image embeddings
print("Generating image embeddings...")
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# ✅ Run inference
print("Running model inference...")
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs["attention_mask"],  # Fix: Ensure attention_mask is correctly passed
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=1024,  # Reduce to prevent excessive memory usage
    do_sample=False,
    use_cache=True
)

# ✅ Decode and display output
print("Decoding response...")
answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print("\n📝 Model Response:\n", answer)

# 🧹 Clear GPU memory after execution
torch.cuda.empty_cache()
