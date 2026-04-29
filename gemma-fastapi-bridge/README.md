# Technical Report: Gemma 4 / OpenVINO Agent Integration

## 1. Project Overview
The objective was to enable `agent.py` (based on the `openai-agents` SDK) to execute SSH commands via Model Context Protocol (MCP) using the experimental **Gemma 4 (E4B)** model on an OpenVINO backend.

## 2. Architecture & Components
To bypass the limitations of the standard OpenVINO Model Server (OVMS) which does not yet support the `gemma4` architecture, we implemented a custom **FastAPI Bridge Server**.

### Key Components:
- **Model:** `OpenVINO/gemma-4-E4B-it-int8-ov`
- **Inference Engine:** `optimum-intel` (custom fork: `rkazants/optimum-intel.git@support_gemma_4`)
- **API Framework:** FastAPI (providing OpenAI-compatible `/v1/chat/completions` endpoint)
- **Tool Calling:** Custom regex-based parser to translate Gemma 4's native `<|tool_call|>` tags into the OpenAI JSON schema.

## 3. Implementation Details

### Dependency Management
The environment was set up using `uv`. Due to strict version pinning in the Gemma 4 fork, dependencies were installed as follows:
- `transformers==5.5.0` (required for the `gemma4` model type)
- `optimum-intel` (from the `support_gemma_4` branch)
- `openvino==2026.1.0`

### Tool-Call Parsing Logic
Gemma 4 uses a specific delimiter-based format for tool calls. We implemented a robust parser in `server.py` to extract these:
- **Format:** `<|tool_call|>call:function_name{key:value}<tool_call|>`
- **Logic:** The server intercepts the model's raw text generation, parses the parameters (handling the `<|"|>` internal quoting used by Gemma 4), and constructs a standard OpenAI `tool_calls` array for the `openai-agents` SDK.

## 4. Troubleshooting & Challenges

### Memory Constraints (OOM)
The primary obstacle encountered was the **Exit Code 137 (OOM)** during the model compilation phase.
- **Hardware:** 16GB RAM (15GB Available).
- **Model Size:** ~8.3GB on disk (INT8 quantization).
- **Observations:** While the model should theoretically fit in memory, the OpenVINO compilation step (converting the IR to an executable CPU graph) causes a significant memory spike that exceeds the 15GB limit.

### Attempted Mitigations:
1. **Thread Limitation:** Set `INFERENCE_NUM_THREADS` and `NUM_STREAMS` to 1 to reduce compilation overhead.
2. **Cache Enablement:** Configured `CACHE_DIR` to minimize redundant compilation.
3. **Direct Loading:** Bypassed `optimum-intel` to load the `openvino_language_model.xml` directly via the OpenVINO Core API.
4. **Swap Expansion:** Attempted to create an 8GB swap file, but was denied due to containerized environment restrictions (`Operation not permitted`).

## 5. Conclusion & Recommendations
The implementation of the API bridge and the custom tool-calling logic is functionally complete. However, the **Gemma-4-E4B** model is currently too resource-intensive for a 16GB RAM environment during the compilation phase.

### To achieve full execution:
1. **Upgrade RAM:** Use an environment with at least 32GB RAM to handle the compilation spike.
2. **Pre-compiled Blobs:** Compile the model on a larger machine and transfer the `ov_cache` directory to the target VM.
3. **Model Quantization:** Explore further quantization or use a smaller Gemma variant (if available) that fits within the memory ceiling.
