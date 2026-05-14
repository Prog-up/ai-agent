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

### Thinking / Reasoning Mode

Pass `"enable_thinking": true` in the request body to activate the model's
reasoning step. When enabled, the model's internal chain of thought is included
in the response wrapped in `<think>...</think>` tags, prepended to the final
answer inside the standard `content` field.

**Non-streaming example response `content`:**
```
<think>
The user is asking about X. Let me reason through...
</think>

The answer is Y.
```

**Streaming:** thinking tokens are streamed inline as `delta.content` chunks,
starting with a `<think>\n` chunk and ending with `\n</think>\n\n` before the
answer begins. Clients that render `<think>` blocks (OpenWebUI, etc.) will
display the reasoning step automatically.

The `enable_thinking` field is silently ignored if `THINKING_SUPPORTED` is
`false` at startup (probe result logged on container start).

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
- **API Surface:** Thinking blocks are embedded in the `content` field wrapped in `<think>...</think>` tags for both streaming and non-streaming responses.

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

1. **Bugs fixed** — `temperature` was silently ignored — all responses were greedy regardless of the parameter. We fixed this by correctly threading `temperature` when `do_sample` is active. Replaced the deprecated `get_event_loop()` with `get_running_loop()`. Fixed a critical issue where `thread.join()` was blocking the async event loop during streaming. Fixed stream `finish_reason` to correctly report `"length"` when `max_tokens` is hit, and `"stop"` otherwise. Improved `parse_thinking` to ensure a robust non-greedy match on reasoning blocks. Fixed `_queue_counter` to appropriately release state dynamically upon stream termination.
2. **Concurrency protection** — Added a strict module-level semaphore (`MAX_CONCURRENT_REQUESTS=1`) initialized via an `asynccontextmanager` lifespan. To prevent indefinite waiting, a queuing boundary (`MAX_QUEUED_REQUESTS=4`) intercepts new queries. Excess requests cleanly return a 503 response envelope:
   `{"error": {"message": "Server at capacity, try again later.", "type": "server_error", "code": "503"}}`
3. **Streaming timing proof** — Token streams arrive individually with discernible delays without blocking the event loop:
   ```
   [1778508383.538] data: {"id": "chatcmpl-fef94c4d86a3", "object": "chat.completion.chunk", "created": 1778508383, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "2,"}, "finish_reason": null}]}
   [1778508384.756] data: {"id": "chatcmpl-fef94c4d86a3", "object": "chat.completion.chunk", "created": 1778508383, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "3,"}, "finish_reason": null}]}
   [1778508385.989] data: {"id": "chatcmpl-fef94c4d86a3", "object": "chat.completion.chunk", "created": 1778508383, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "4,"}, "finish_reason": null}]}
   [1778508387.224] data: {"id": "chatcmpl-fef94c4d86a3", "object": "chat.completion.chunk", "created": 1778508383, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "5,"}, "finish_reason": null}]}
   [1778508388.460] data: {"id": "chatcmpl-fef94c4d86a3", "object": "chat.completion.chunk", "created": 1778508383, "model": "OpenVINO/gemma-4-E4B-it-int8-ov", "choices": [{"index": 0, "delta": {"content": "6,"}, "finish_reason": null}]}
   ```
4. **New parameters** — Added `top_p`, `top_k`, and `repetition_penalty` — which are fully passed down into the OpenVINO layer and applied dynamically. Added `stop` sequence functionality, implemented manually as best-effort post-hoc truncation applied immediately across text iterations. Rejection gating is active for `n != 1`.
5. **Metrics** — The `/metrics` endpoint is instrumented via `prometheus-fastapi-instrumentator`.
   ```
   # HELP python_gc_objects_collected_total Objects collected during gc
   # TYPE python_gc_objects_collected_total counter
   python_gc_objects_collected_total{generation="0"} 13565.0
   python_gc_objects_collected_total{generation="1"} 2333.0
   python_gc_objects_collected_total{generation="2"} 257.0
   ```
6. **Dockerfile pin** — We pinned the optimum branch pointer on `2026-05-11`. The explicit SHA utilized is `eac389347523177511abe37908090d9e5c12e714` protecting downstream containers from unexpected upstream logic transitions.
7. **What was not changed** — `context_length` dynamically checks `model_max_length` but applies a 128k safety barrier in instances where the default configurations contain placeholders (e.g. `1000000000000000019884624838656`) avoiding unexpected behavior limits across API consumers.

## Hermes CLI Compatibility

1. **Configuration**
   The following configuration block works natively with Hermes using the `custom` provider.
   ```yaml
   # ~/.hermes/config.yaml
   default_provider: local-ovms
   default_model: "OpenVINO/gemma-4-E4B-it-int8-ov"

   providers:
     local-ovms:
       type: openai_compatible
       base_url: "http://localhost:8000/v1"
       api_key: "not-needed"
       models:
         - id: "OpenVINO/gemma-4-E4B-it-int8-ov"
           context_length: 131072

   display:
     show_cost: false

   auxiliary:
     compression:
       model: "OpenVINO/gemma-4-E4B-it-int8-ov"
   ```

2. **Feature compatibility table**

| Feature | Status | Notes |
|---|---|---|
| Single query (`-q`) | ✅ | Fully operational |
| Streaming | ✅ | Timings verified; progressive token delivery |
| Token count in status bar | ✅ | Supported via `stream_options.include_usage` |
| `/reasoning high` (thinking) | ✅ | Triggers `<think>` block generation |
| `/personality` | ✅ | Supported |
| Multi-turn context | ✅ | Supported |
| `/usage` | ✅ | Usage stats successfully report |
| Session resume (`-c`, `-r`) | ✅ | Operational |
| `/background` | ✅ | Background tasks successfully return |
| Image input (vision) | ✅ | Data URI base64 images properly decoded |
| `/compress` | ✅ | Context compression fully operational |

3. **Server.py changes made**
   - Mapped `reasoning_effort="high"` to trigger `enable_thinking=True` for reasoning support.
   - Added `_decode_image_url` helper in `_prepare_inputs` to intercept `image_url` payloads (data URI format) and seamlessly convert them into PIL format for OpenVINO inputs.
   - Set `model_config = ConfigDict(extra='ignore')` inside Pydantic schemas (e.g. `ChatCompletionRequest`) to prevent standard Hermes client headers (`presence_penalty`, `seed`, `user`) from throwing `422 Unprocessable Entity`.
   - Included `ResponseFormat` schema fallback handler defaulting un-supported payloads to raw text logic.
   - Verified `/v1/models` strictly outputs `context_length` metric accurately readable by Hermes context boundaries.

4. **Known limitations**
   - The OpenVINO server triggers a `RuntimeError: Infer Request is busy` if the `gen_thread` does not successfully process early thread disconnects. A rigid `finally` catch prevents the API from totally halting, yet aggressive retries from Hermes can periodically provoke it.

5. **Quick start**
   ```bash
   # Add the config locally
   hermes config set model.provider custom
   hermes config set model.default OpenVINO/gemma-4-E4B-it-int8-ov
   hermes config set model.base_url http://localhost:8000/v1
   
   # Confirm operational integrity
   hermes chat -q "Say exactly: OK"
   ```

## v1.3 — Hermes Compatibility Fixes

1. **Bugs fixed**
   - `_decode_image_url`: Hermes sends images as data URIs; the server previously crashed, so we added base64 to PIL decoding.
   - `reasoning_effort`: Hermes uses this field for `/reasoning high`; we added it to the schema and mapped it to `enable_thinking`.
   - `ResponseFormat`: Hermes occasionally requests `{"type": "text"}`; added a schema to accept it and warn if JSON mode is asked for.
   - `ConfigDict extra ignore`: Hermes sends headers like `presence_penalty` and `user`; explicitly ignoring extra fields prevents HTTP 422 errors.
   - `DEBUG_LOG_REQUESTS`: Debug middleware flooded logs with base64 image data; it is now gated behind an environment variable.
   - `_model_lock`: The extra lock falsely implied multi-threading protection when the semaphore already guarantees it; it was removed.
   - `finally: pass`: An empty, useless block wrapping the generator was removed for code cleanliness.
   - `system_fingerprint` & `max_context_length`: The `/v1/models` endpoint lacked these; we added them for Hermes to accurately gauge token capacity.

2. **Corrected Feature compatibility table**

| Feature | Status | Notes |
|---|---|---|
| Single query (`-q`) | ✅ | Fully operational. |
| Streaming | ✅ | Timings verified; progressive token delivery (no batching). |
| Token count in status bar | ✅ | Supported via `stream_options.include_usage`. |
| `/reasoning high` (thinking) | ✅ | Triggers `<think>` block generation natively. |
| `/personality` | ✅ | Supported. |
| Multi-turn context | ✅ | Supported. |
| `/usage` | ✅ | Usage stats successfully reported. |
| Session resume (`-c`, `-r`) | ✅ | Operational. |
| `/background` | ✅ | Background tasks successfully return. |
| Image input (vision) | ✅ | Data URI base64 images properly decoded. |
| `/compress` | ✅ | Context compression fully operational. |

3. **Image Input Testing**
   Image input successfully identified the color. Example response: `'The color of the square is blue.'`

4. **Reasoning Mode Testing**
   `/reasoning high` produced a visible `<think>` block. Example first 50 characters: `<think>\nThe user wants to know the product of 13 `
