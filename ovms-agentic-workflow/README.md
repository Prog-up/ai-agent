---
license: apache-2.0
base_model:
  - google/gemma-4-E4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- gemma4
- conversational
library_name: openvino
---

# gemma-4-E4B-it-int8-ov

* Model creator: [google](https://huggingface.co/google)
* Original model: [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)

> [!WARNING]
> **EXPERIMENTAL MODEL**
> This model has not been fully validated with OpenVINO and is using a custom branch of optimum-intel. It may be fully supported and validated in the future.

## Description

This is [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2026/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT8 by [NNCF](https://github.com/openvinotoolkit/nncf).

## Quantization Parameters

Weight compression was performed using `nncf.compress_weights` with the following parameters:

* mode: **INT8_ASYM**
* group_size: **-1**

For more information on quantization, check the [OpenVINO model optimization guide](https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html).

## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO version 2026.1.0 and higher
* Optimum Intel 1.27.0 and higher

## Running Model Inference with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)

1. Install packages required for using Optimum Intel:

```
pip install "git+https://github.com/rkazants/optimum-intel.git@support_gemma_4" --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers==5.5.0
pip install torchvision Pillow --extra-index-url https://download.pytorch.org/whl/cpu
```

2. Run model inference:

```python
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
```

You can find more detailed usage examples in OpenVINO Notebooks:

- [Gemma 4](https://openvinotoolkit.github.io/openvino_notebooks/?search=Gemma+4)

## Limitations

Check the original [model card](https://huggingface.co/google/gemma-4-E4B-it) for limitations.

## Legal information

The original model is distributed under [Apache License Version 2.0](https://huggingface.co/google/gemma-4-E4B-it/blob/main/LICENSE) license. More details can be found in [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or
contributing to adverse impacts on human rights. See
[Intel's Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf).
Intel's products and software are intended only to be used in applications
that do not cause or contribute to adverse impacts on human rights.
