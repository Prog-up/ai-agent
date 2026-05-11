# Gemma 4 on OpenVINO with OpenAI-Compatible API

This project provides a custom inference container for running the **Gemma 4** model (`OpenVINO/gemma-4-E4B-it-int8-ov`) using OpenVINO and exposing an OpenAI-compatible API.

## Why a Custom Container?

The official `openvino/model_server:weekly` (version 2026.2.0) does **not** support Gemma 4 yet. Attempting to load the model into OVMS results in the following error:

```
[serving][error][servable_initializer.cpp:214] Error during llm node initialization for models_path: /models/OpenVINO/gemma-4-E4B-it-int8-ov/./ exception: Exception from ../../../../../repos/openvino.genai/src/cpp/src/visual_language/vlm_config.cpp:34:
Unsupported 'gemma4' VLM model type
```

Specifically, Gemma 4 is a Visual Causal LM that requires a custom branch of `optimum-intel` (`support_gemma_4`) and `transformers==5.5.0` which are not yet integrated into the standard OVMS pipeline.

## Environment & Hardware

The following environment was used for development and benchmarking:

- **CPU:** Intel(R) Xeon(R) CPU E5-2643 v2 @ 3.50GHz (2 Sockets, 6 Cores/Socket, 2 Threads/Core = 24 vCPUs)
- **RAM:** 122 GiB
- **OS:** Ubuntu 22.04 (in a Docker container)
- **Docker:** 29.2.1
- **Python:** 3.10.x (inside container)
- **OpenVINO:** 2026.1.0

## Model Details

- **Model ID:** `OpenVINO/gemma-4-E4B-it-int8-ov`
- **Format:** OpenVINO IR (INT8 Quantized)
- **Size on disk:** 7.8 GB
- **Memory usage:** Requires ~27 GB RAM for stable loading and inference.

## Architecture

```
User → OpenAI Client → FastAPI Server (port 8000) → OVModelForVisualCausalLM → OpenVINO Runtime → Model files
```

The server is built with FastAPI and Uvicorn, using a `ThreadPoolExecutor` to handle blocking inference calls without stalling the event loop.

## API Reference

### Health Check
`GET /health`
Returns `{"status": "ok", "model": "OpenVINO/gemma-4-E4B-it-int8-ov"}`

### List Models
`GET /v1/models`
Returns the model metadata in OpenAI format.

### Chat Completions
`POST /v1/chat/completions`
Supports both streaming and non-streaming requests.

**Example Request:**
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenVINO/gemma-4-E4B-it-int8-ov",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Benchmark Results

| Test Case | Latency | Input Tokens | Output Tokens | Throughput |
|-----------|---------|--------------|---------------|------------|
| Short response | 1.61s | 12 | 2 | 1.2 tok/s |
| Medium response| 34.46s | 19 | 78 | 2.3 tok/s |
| Long response  | 47.17s | 16 | 110 | 2.3 tok/s |

- **Mean Throughput:** ~1.9 tokens/sec
- **Resource Usage:** ~28 GB RAM, ~100% CPU (scaled by thread count) during inference.

## Optimization Notes

- **Thread Tuning:** Setting `OV_CPU_BACKEND_NUM_THREADS` to the physical core count (12) provided the best balance. Setting it to 24 (all vCPUs) slightly decreased throughput due to context switching.
- **Async Execution:** Using a dedicated `ThreadPoolExecutor` in the FastAPI server ensures that multiple concurrent requests don't block the API, although actual inference is sequential on a single CPU device.

## v1.1 — Streaming, Thinking & GPU Detection

### What was broken and why
The previous version used "fake streaming" by splitting a fully generated response into words. This resulted in high TTFT (Time To First Token) and a poor user experience as the client received nothing until the entire generation was complete.

### Real Streaming Implementation
Implemented true token streaming using the `transformers.TextIteratorStreamer` class.
- **Background Thread:** The `model.generate` call runs in a dedicated `threading.Thread`, pushing tokens into the streamer's queue.
- **Asynchronous Yielding:** The FastAPI server iterates over the streamer and yields tokens to the client as they arrive, significantly reducing TTFT.
- **Special Tokens:** Set `skip_special_tokens=False` to ensure thinking tags (`<|channel>thought`) are captured and processed.

### Thinking Support
Gemma 4 Instruct's native reasoning mode was enabled and exposed.
- **Activation:** Passing `enable_thinking=True` to `apply_chat_template` inserts the `<|think|>` token into the system prompt.
- **Parsing:** Added a robust `parse_thinking` function that identifies the `<|channel>thought` start tag and the `<channel|>` end tag.
- **API Surface:** Thinking blocks are surfaced via a dedicated `thinking` field in non-streaming responses and `delta.thinking` in streaming chunks.

### GPU Detection & Precision
- **Auto-detection:** Added logic using `openvino.Core().available_devices` to automatically target `GPU` if available, falling back to `CPU`.
- **Override:** The `DEVICE` environment variable can be used to force a specific device.
- **Precision:** Enabled `INFERENCE_PRECISION_HINT: f32` in the `ov_config` to ensure high-quality output on varied hardware.

### Observations on this Machine
- **Available devices:** `['CPU']`
- **Using device:** `CPU`
- **Performance:** Mean throughput remained stable at ~2.3 tok/s with `f32` precision hint and real streaming enabled.
- **Memory:** Increased Docker memory allocation to 64GB to handle the increased overhead of the reasoning blocks and precision hint.

## Reproduction Steps

1. **Clone and Setup:**
   ```bash
   mkdir -p /root/gemma4-ovms
   cd /root/gemma4-ovms
   ```

2. **Download Model:**
   ```bash
   pip3 install huggingface_hub
   python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='OpenVINO/gemma-4-E4B-it-int8-ov', local_dir='/opt/models/gemma-4-E4B-it-int8-ov')"
   ```

3. **Build Image:**
   ```bash
   docker build -t gemma4-ovms:latest .
   ```

4. **Run Server:**
   ```bash
   docker run -d \
     --name gemma4-api \
     -p 8000:8000 \
     -v /opt/models/gemma-4-E4B-it-int8-ov:/models:ro \
     --memory="64g" \
     -e OV_CPU_BACKEND_NUM_THREADS=12 \
     gemma4-ovms:latest
   ```

5. **Verify:**
   ```bash
   curl http://localhost:8000/health
   ```

## v1.2 — Stability, Correctness & Observability

1. **Bugs fixed** — `temperature` was silently ignored, making all responses greedy regardless of the parameter. Replaced the deprecated `get_event_loop()` with `get_running_loop()`. Fixed a critical issue where `thread.join()` was blocking the async event loop. Improved the regex for parsing the thinking block to handle first-match non-greedy parsing properly. Fixed stream `finish_reason` to correctly report `"length"` when `max_tokens` is hit.
2. **Concurrency protection** — Implemented a semaphore (`MAX_CONCURRENT_REQUESTS=1`) to protect OpenVINO state and a queue limit (`MAX_QUEUED_REQUESTS=4`). Excess requests get a clean 503 response:
   `{"error": {"message": "Server at capacity, try again later.", "type": "server_error", "code": "503"}}`
3. **Streaming timing proof** — Verified real inter-token gaps in streaming:
   ```
   [1778507173.538] data: {"id": "chatcmpl-1f08e5df1a78", "object": "chat.completion.chunk", "created": 1778507170, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "2,"}, "finish_reason": null}]}
   [1778507174.756] data: {"id": "chatcmpl-1f08e5df1a78", "object": "chat.completion.chunk", "created": 1778507170, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "3,"}, "finish_reason": null}]}
   [1778507175.989] data: {"id": "chatcmpl-1f08e5df1a78", "object": "chat.completion.chunk", "created": 1778507170, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "4,"}, "finish_reason": null}]}
   [1778507177.224] data: {"id": "chatcmpl-1f08e5df1a78", "object": "chat.completion.chunk", "created": 1778507170, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "5,"}, "finish_reason": null}]}
   [1778507178.460] data: {"id": "chatcmpl-1f08e5df1a78", "object": "chat.completion.chunk", "created": 1778507170, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "6,"}, "finish_reason": null}]}
   ```
4. **New parameters** — Added `top_p`, `top_k`, and `repetition_penalty`, which are fully enforced at the generation level. `stop` strings are applied via best-effort post-hoc truncation. `n` is explicitly limited to `1` with early rejection.
5. **Metrics** — Integrated `/metrics` for observability:
   ```
   # HELP python_gc_objects_collected_total Objects collected during gc
   # TYPE python_gc_objects_collected_total counter
   python_gc_objects_collected_total{generation="0"} 13565.0
   python_gc_objects_collected_total{generation="1"} 2333.0
   python_gc_objects_collected_total{generation="2"} 257.0
   ```
6. **Dockerfile pin** — Pinned `support_gemma_4` to `eac389347523177511abe37908090d9e5c12e714` for guaranteed reproducibility.
7. **What was not changed** — All planned changes behaved as expected. Note that `context_length` dynamically reads from the processor, yielding the large placeholder `1000000000000000019884624838656` as provided by the model config, which was left un-altered.
