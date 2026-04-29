from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
import torch

MODEL_ID = "OpenVINO/gemma-4-E4B-it-int8-ov"
print(f"Loading processor {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Loading model {MODEL_ID}...")
model = OVModelForVisualCausalLM.from_pretrained(MODEL_ID, device="CPU")
print("Model loaded successfully.")

messages = [{"role": "user", "content": "Hello"}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(outputs[0], skip_special_tokens=False))
