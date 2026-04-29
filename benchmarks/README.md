# AI Agent Benchmarks

This directory contains various benchmarks conducted during the research phase of the AI agent development.

## Sub-projects

- **[local-vs-openai](./local-vs-openai)**: Comparison between local OpenVINO inference and OpenAI API server (e.g., OVMS or vLLM).
- **[ov-vs-llamacpp](./ov-vs-llamacpp)**: Detailed comparison between OpenVINO GenAI and llama-cpp-python using various models (Mistral, Mixtral).
- **[deepseek-ov](./deepseek-ov)**: Specialized benchmarks for DeepSeek-V2-Lite converted to OpenVINO INT4.

## Setup

It is recommended to use `uv` for managing dependencies.

```bash
uv sync
```

## Running Benchmarks

Navigate to the specific sub-directory and run the benchmark scripts.
