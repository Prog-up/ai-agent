# Gemma 4 Docker MCP Gateway

Research on integrating Gemma 4 with the Docker MCP Toolkit Gateway.

## Files

- **[docker-mcp.md](./docker-mcp.md)**: Main research notes on Docker MCP.
- **[gemma_mcp.py](./gemma_mcp.py)**: Implementation of Gemma 4 with MCP.
- **[gemma_mcp_time.py](./gemma_mcp_time.py)**: Performance/timing analysis.
- **[Gemma4.md](./Gemma4.md)**: Specific notes on Gemma 4 (E4B) technical specs.
- **[mcp-client.md](./mcp-client.md)**: Documentation for the MCP client integration.
- **[report.md](./report.md)**: Final research report.

## Key Findings

- Gemma 4 (E4B) requires `transformers==5.5.0`.
- Memory management is critical (OOM issues during compilation).
- Using `INFERENCE_NUM_THREADS=1` and `CACHE_DIR` helps stability on 16GB RAM.
