import time
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("mistral_ov", "GPU")

token_count = 0

def callback(token):
    global token_count
    token_count += 1
    return False

start = time.time()

pipe.generate(
    "Explain OpenVINO in one paragraph.",
    max_new_tokens=200,
    streamer=callback
)

end = time.time()

print("Tokens:", token_count)
print("Tokens/sec:", token_count / (end - start))
