# Technical Report: Agentic SSH Management with OVMS and MCP

## 1. Executive Summary
This session focused on the integration of the **OpenAI Agents SDK** with **Intel OpenVINO™ Model Server (OVMS)** and the **Model Context Protocol (MCP)**. We successfully developed a chatbot capable of managing remote servers via SSH using natural language, hosted on an Intel-accelerated infrastructure.

## 2. Architectural Pillars

### A. The Logic Layer: OpenAI Agents SDK
We utilized the `openai-agents` Python SDK. Unlike a standard LLM call, this SDK manages a "Runner" loop that handles:
- **Handshakes:** Automatically discovering tools from MCP servers.
- **Interception:** Capturing tool-call requests from the LLM.
- **Execution:** Running tools and feeding results back into the conversation history.

### B. The Inference Layer: OpenVINO Model Server (OVMS)
OVMS acts as the OpenAI-compatible backend. We leveraged the **GenAI Backend**, which is optimized for Large Language Models (LLMs) and supports:
- **Continuous Batching & Paged Attention:** For high-throughput inference.
- **Standardized Endpoints:** Providing `/v3/chat/completions` for modern agent workflows.

### C. The Tooling Layer: Model Context Protocol (MCP)
MCP allows the agent to interact with the world. We integrated an **SSH MCP Server** (`@fangjunjie/ssh-mcp-server`) which exposes capabilities like `run-command` to the agent.

## 3. Technical Discoveries & Hurdles

### The Gemma 4 Architecture Gap
While attempting to use the **Gemma-4-E4B-it** model, we encountered a critical failure: `Unsupported 'gemma4' VLM model type`. 
- **Discovery:** Support for Gemma 4 is currently experimental and exists in a specific Python fork of `optimum-intel`.
- **Finding:** The OVMS C++ backend (`openvino-genai`) does not yet include the architecture mappings for the Gemma 4 VLM projection, making it incompatible with the high-performance `text_generation` task in the standard Docker image.

### OVMS Configuration Nuances
Setting up OVMS for local LLMs required solving several configuration challenges:
1. **Directory Structure:** OVMS expects models to be versioned. Model files must be placed in a subdirectory named `1/` (e.g., `models/model_name/1/`).
2. **Task Definition:** The `--task text_generation` flag is mandatory to trigger the GenAI pipeline. Without it, the server defaults to standard vision/classification logic.
3. **The Tool Parser (Crucial):** By default, many LLMs do not output tool calls in the exact JSON format the SDK expects. We discovered that enabling `--tool_parser hermes3` in the OVMS configuration is required to correctly map model outputs to the OpenAI tool-calling schema.

## 4. Work Performed

### Development of `agent.py`
We created a robust, interactive chatbot script that:
- Connects to a local/remote SSH MCP server.
- Supports single-query execution via CLI (`--query`) and interactive chat loops.
- Implements real-time token streaming for a better user experience.

### Docker Orchestration
We successfully launched a stable OVMS environment using **Qwen 2.5** (aliased as `gemma-4-it`). 
**Final Working Command:**
```bash
docker run -d --name ovms-gemma \
  -p 8000:8000 \
  -v $(pwd)/models:/models:rw \
  openvino/model_server:latest \
  --source_model OpenVINO/Qwen2.5-7B-Instruct-int4-ov \
  --model_repository_path /models \
  --task text_generation \
  --model_name gemma-4-it \
  --rest_port 8000 \
  --tool_parser hermes3
```

## 5. Verification & Validation
We performed an end-to-end test of the system:
1. **Input:** "Run the bash command 'touch /root/qwen_test_file' on the remote server."
2. **Agent Action:** The agent recognized the need for the `execute-command` tool and triggered it via the SSH MCP server.
3. **Outcome:** A direct SSH check confirmed the file was created:
   `ls -l /root/qwen_test_file` -> `Successfully verified`.

## 6. Conclusion
The combination of `openai-agents` and `OVMS` provides a powerful, local alternative to cloud-based agentic services. While experimental models like Gemma 4 require more mature backend support, standard models like Qwen 2.5 prove that the ecosystem is ready for production-grade, local tool-augmented AI agents.
