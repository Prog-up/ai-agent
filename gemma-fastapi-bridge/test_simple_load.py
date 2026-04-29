from optimum.intel.openvino import OVModelForVisualCausalLM
import time

MODEL_ID = "OpenVINO/gemma-4-E4B-it-int8-ov"
print(f"Loading model {MODEL_ID}...")
model = OVModelForVisualCausalLM.from_pretrained(MODEL_ID, device="CPU")
print("Model loaded successfully.")
time.sleep(10)
