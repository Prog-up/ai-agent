import openvino as ov
import os
from huggingface_hub import snapshot_download

model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
model_dir = snapshot_download(model_id)
model_xml = os.path.join(model_dir, "openvino_language_model.xml")

core = ov.Core()
core.set_property("CPU", {"CACHE_DIR": "ov_cache"})

print("Reading model...")
model = core.read_model(model_xml)

print("Compiling model...")
# Minimal config
config = {
    "INFERENCE_NUM_THREADS": "1",
    "NUM_STREAMS": "1",
}
compiled_model = core.compile_model(model, "CPU", config)
print("Success!")
