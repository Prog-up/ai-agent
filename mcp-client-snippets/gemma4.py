from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
from PIL import Image
import requests

model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
processor = AutoProcessor.from_pretrained(model_id)
model = OVModelForVisualCausalLM.from_pretrained(model_id)

url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is unusual in this picture?"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt")
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
response = processor.decode(output[0][input_len:], skip_special_tokens=True)
print(response)