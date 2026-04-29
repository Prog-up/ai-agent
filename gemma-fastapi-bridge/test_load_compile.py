from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
import torch
import time

MODEL_ID = "OpenVINO/gemma-4-E4B-it-int8-ov"
print(f"Loading processor {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Loading model {MODEL_ID} (compile=False)...")
model = OVModelForVisualCausalLM.from_pretrained(MODEL_ID, device="CPU", compile=False)
print("Model loaded (not compiled). Compiling...")
model.compile()
print("Model compiled successfully.")

messages = [{"role": "user", "content": "Hello"}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(prompt, return_tensors="pt")
print("Generating...")
outputs = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(outputs[0], skip_special_tokens=False))
