from transformers import AutoProcessor
import json

model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
processor = AutoProcessor.from_pretrained(model_id)
print(f"Chat template: {processor.chat_template}")
