import openvino as ov
import os
from huggingface_hub import snapshot_download

model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
model_dir = snapshot_download(model_id)
model_xml = os.path.join(model_dir, "openvino_language_model.xml")
model_bin = os.path.join(model_dir, "openvino_language_model.bin")

core = ov.Core()
core.set_property("CPU", {"CACHE_DIR": "ov_cache"})
print("Reading model...")
model = core.read_model(model=model_xml, weights=model_bin)
print("Compiling model...")
# Try to limit memory usage during compilation
compiled_model = core.compile_model(model, "CPU", {"CPU_THREADS_NUM": "1"})
print("Model compiled successfully.")
