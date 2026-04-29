import openvino_genai as ov_genai
import time

def main():
    print("Hello from benchmark!")

if __name__ == "__main__":
    main()

    pipe = ov_genai.LLMPipeline("mistral_ov", "GPU")

    prompt = "Explain what OpenVINO is in one paragraph."

    start = time.time()

    result = pipe.generate(
        prompt,
        max_new_tokens=200
    )

    end = time.time()

    tokens_generated = len(result.tokens)
    duration = end - start

    print("Generated tokens:", tokens_generated)
    print("Time:", duration)
    print("Tokens/sec:", tokens_generated / duration)
