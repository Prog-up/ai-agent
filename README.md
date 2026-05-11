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
     --memory="48g" \
     -e OV_CPU_BACKEND_NUM_THREADS=12 \
     gemma4-ovms:latest
   ```

5. **Verify:**
   ```bash
   curl http://localhost:8000/health
   ```
