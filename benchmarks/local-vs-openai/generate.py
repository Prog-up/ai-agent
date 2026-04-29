from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    # model="OpenVINO/Qwen3-8B-int4-ov",
    model="OpenVINO/mistral-7b-instruct-v0.1-int8-ov",
    messages=[#{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "What are the 3 main tourist attractions in Paris?"}
    ],
    max_tokens=3000,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
