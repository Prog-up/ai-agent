import openvino as ov
import numpy as np

model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
core = ov.Core()
print("Reading model...")
# The model files are likely in ~/.cache/huggingface/hub/models--OpenVINO--gemma-4-E4B-it-int8-ov/snapshots/...
# But I can use the local path if I know where it is.
# optimum-intel downloaded it to some place.

import os
from huggingface_hub import snapshot_download
model_dir = snapshot_download(model_id)
print(f"Model dir: {model_dir}")

model_xml = os.path.join(model_dir, "openvino_language_model.xml")
print(f"Reading {model_xml}...")
model = core.read_model(model_xml)
print("Compiling model...")
compiled_model = core.compile_model(model, "CPU")
print("Model compiled successfully.")
