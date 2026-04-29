import time
from openai import OpenAI
import openvino_genai as ov_genai

# =========================
# CONFIG
# =========================

USE_OPENAI_SERVER = True  # False = local OpenVINO

# ---- Model selection ----
# OpenVINO local
LOCAL_MODEL = "mistral_ov"

# OpenAI-compatible server models
# MODEL_NAME = "OpenVINO/Qwen3-8B-int4-ov"
MODEL_NAME = "OpenVINO/mistral-7b-instruct-v0.1-int8-ov"

PROMPT = "What are the 3 main tourist attractions in Paris?"
MAX_TOKENS = 300


# =========================
# METHOD 1 — OpenVINO local
# =========================

def run_local():
    pipe = ov_genai.LLMPipeline(LOCAL_MODEL, "GPU")

    token_count = 0

    def callback(token):
        nonlocal token_count
        token_count += 1
        return False

    start = time.time()

    pipe.generate(
        PROMPT,
        max_new_tokens=MAX_TOKENS,
        streamer=callback
    )

    end = time.time()

    print("\n--- LOCAL OPENVINO ---")
    print("Tokens:", token_count)
    print("Time:", end - start)
    print("Tokens/sec:", token_count / (end - start))


# =========================
# METHOD 2 — OpenAI server
# =========================

def run_openai():
    client = OpenAI(
        base_url="http://localhost:8000/v3",
        api_key="unused"
    )

    token_count = 0

    start = time.time()

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        max_tokens=MAX_TOKENS,
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

            # crude token estimation (1 token ≈ 4 chars)
            token_count += max(1, len(content) // 4)

    end = time.time()

    print("\n--- OPENAI SERVER ---")
    print("Estimated tokens:", token_count)
    print("Time:", end - start)
    print("Tokens/sec:", token_count / (end - start))


# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    if USE_OPENAI_SERVER:
        run_openai()
    else:
        run_local()
