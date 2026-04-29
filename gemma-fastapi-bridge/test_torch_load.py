from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import time

MODEL_ID = "OpenVINO/gemma-4-E4B-it-int8-ov"
print(f"Loading model {MODEL_ID} with torch...")
# Note: This is an OpenVINO model, so AutoModelForCausalLM won't work directly if it's ONLY OV.
# But sometimes they have both.
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    print("Model loaded with torch successfully.")
except Exception as e:
    print(f"Failed to load with torch: {e}")
