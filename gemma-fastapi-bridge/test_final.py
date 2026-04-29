import openvino as ov
import os
from huggingface_hub import snapshot_download
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
import torch

# Limit threads
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
print("Loading model...")

ov_config = {
    "INFERENCE_NUM_THREADS": "1",
    "NUM_STREAMS": "1",
}

try:
    processor = AutoProcessor.from_pretrained(model_id)
    model = OVModelForVisualCausalLM.from_pretrained(
        model_id,
        device="CPU",
        ov_config=ov_config,
        compile=True
    )
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
