import openvino as ov
import os
from huggingface_hub import snapshot_download

model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
model_dir = snapshot_download(model_id)
model_xml = os.path.join(model_dir, "openvino_language_model.xml")
model_bin = os.path.join(model_dir, "openvino_language_model.bin")

core = ov.Core()
print("Reading model...")
model = core.read_model(model=model_xml, weights=model_bin)
print("Compiling model...")
compiled_model = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 1, "NUM_STREAMS": 1})
print("Model compiled successfully.")
