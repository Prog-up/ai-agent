import time
import openvino_genai as ov_genai
from transformers import AutoTokenizer

model_dir = "mistral_ov"

pipe = ov_genai.LLMPipeline(model_dir, "GPU")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "Explain what OpenVINO is."

start = time.time()

output = pipe.generate(
    prompt,
    max_new_tokens=200
)

end = time.time()

# count tokens
input_tokens = len(tokenizer.encode(prompt))
output_tokens = len(tokenizer.encode(output))
generated_tokens = output_tokens - input_tokens

duration = end - start

print("Generated tokens:", generated_tokens)
print("Time:", duration)
print("Tokens/sec:", generated_tokens / duration)
