# OpenAI Agent with OVMS and MCP Support

## 1. Overview
`openai_agent.py` is a specialized implementation of an AI agent that leverages Intel's hardware acceleration via **OpenVINO™ Model Server (OVMS)**. It uses the **Model Context Protocol (MCP)** to connect to external tools, allowing the agent to perform actions like querying weather data or interacting with a local filesystem.

## 2. Architecture & Tech Stack
The script functions as an orchestrator between three main layers:

*   **Inference Layer (OVMS):** Serves Large Language Models (LLMs) optimized for Intel hardware (CPU, GPU, NPU). It provides an OpenAI-compatible API endpoint.
*   **Logic Layer (OpenAI Agents SDK):** Uses the `openai-agents` library to manage the agent's decision-making loop, tool calling, and state management.
*   **Tooling Layer (MCP):** Uses the Model Context Protocol to provide standardized access to external tools via **SSE (Server-Sent Events)** or **Stdio**.

## 3. Key Components

### `openai-agents` SDK
The script imports `Agent`, `Runner`, and `RunConfig` from the `agents` package (`openai-agents` in `requirements.txt`). This SDK provides:
- **Agent Orchestration:** Manages how the LLM interacts with tools.
- **Streaming Support:** Implements real-time response generation using `Runner.run_streamed`.
- **Provider Abstraction:** Uses a `ModelProvider` class to bridge different LLM backends.

### OpenVINO Model Server (OVMS) integration
The class `OVMSModelProvider` defines how the agent communicates with the LLM.
- **Base URL:** Typically points to an OVMS instance (default: `http://localhost:8000/v3`).
- **Compatibility:** Since OVMS is OpenAI-compatible, the agent uses `AsyncOpenAI` client to send requests.

### Model Context Protocol (MCP)
The agent is "augmented" with tools via MCP servers. The script supports two types of connections:
1.  **`MCPServerStdio`**: Used for the **Filesystem MCP Server**. It launches a local process (via `npx`) and communicates through standard input/output.
2.  **`MCPServerSse`**: Used for the **Weather MCP Server** (on non-Windows systems). It connects to a remote or local server via a web URL.

## 4. MCP Client Implementation Details

The MCP client in this code is implemented using native classes from the `agents.mcp` module. Unlike generic MCP clients that require manual tool definition, this implementation is **schema-driven**:

*   **Connection Lifecycle:** The script manually manages connections by calling `await server.connect()` for each server in the `mcp_servers` list before starting the agent run.
*   **Encapsulation:** The `Agent` class constructor accepts a list of `mcp_servers`. The SDK then handles the "handshake" where it queries the MCP server for its available tools and their JSON schemas.
*   **Dynamic Tool Injection:** When the LLM is queried, the SDK automatically includes the tools discovered from the MCP servers in the OpenAI-compatible `tools` parameter of the API call.
*   **Parameter Passing:**
    *   For `MCPServerStdio`, it passes `command`, `args`, and `env` (to handle proxies).
    *   For `MCPServerSse`, it simply takes a `url`.

## 5. Relationship with Official SDK

The code has a direct dependency on the **[openai/openai-agents-python](https://github.com/openai/openai-agents-python)** library (installed as `openai-agents` in `requirements.txt`).

*   **Native Integration:** The `MCPServerSse` and `MCPServerStdio` classes are part of the core `openai-agents` SDK. This means the MCP protocol is a "first-class citizen" in the OpenAI Agents framework.
*   **Standard Compliance:** By using this SDK, the code follows the official OpenAI patterns for tool calling, ensuring that even when running on local Intel hardware via OVMS, the agent behaves identically to one running on OpenAI's cloud infrastructure.

## 6. Steps to Implement MCP Support (Code Breakdown)

To support MCP in this architecture, the code follows these specific steps:

### Step 1: Import MCP Modules
The SDK provides dedicated classes for different MCP transport protocols.
```python
from agents.mcp import MCPServerSse, MCPServerStdio
```

### Step 2: Define Server Configuration
The code sets up the parameters for the servers. For local tools (like the filesystem), it uses shell commands. For remote tools, it uses URLs.
```python
# example for Stdio
fs_params = {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}

# example for SSE
weather_params = {"url": "http://localhost:8080/sse"}
```

### Step 3: Instantiate and Collect Servers
Server objects are created and stored in a list. This allows the agent to use multiple tool sources simultaneously.
```python
mcp_servers = []
mcp_servers.append(MCPServerStdio(name="FS", params=fs_params))
mcp_servers.append(MCPServerSse(name="Weather", params=weather_params))
```

### Step 4: Register Servers with the Agent
The `Agent` constructor is designed to natively accept the `mcp_servers` list.
```python
agent = Agent(
    name="Assistant",
    mcp_servers=mcp_servers,
    # ... other settings
)
```

### Step 5: Establish Connections
Before calling the `Runner`, every MCP server must be explicitly connected. This is an asynchronous operation where the agent performs the initial handshake and tool discovery.
```python
for server in agent.mcp_servers:
    await server.connect()
```

### Step 6: Execute the Agent Loop
Once connected, the `Runner` handles the complexity of detecting when the LLM wants to use a tool, executing the MCP call, and feeding the result back into the conversation.

## 7. The Tool Calling Feedback Loop

One of the most complex parts of an agent is how it bridges the gap between the LLM's text output and the actual execution of a tool. Here is the step-by-step process of how the LLM "calls" an MCP tool and "sees" the result:

### 1. Tool Exposure (The Handshake)
When `await server.connect()` is called, the SDK queries the MCP server for a list of its available tools. This returns a set of **JSON Schemas** (describing function names, descriptions, and expected arguments). These schemas are then injected into the system prompt of the LLM.

### 2. The Decision (Reasoning)
The LLM (running on OVMS) receives the user query and the tool definitions. If it decides a tool is needed, it doesn't just output text; it generates a specific **`tool_call`** structure.
*   *Example:* `{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}`

### 3. Interception & Execution (The Middleman)
The `openai-agents` SDK's **`Runner`** intercepts this `tool_call` before it reaches the user. 
- It identifies that `read_file` belongs to the `FS MCP Server`.
- It translates the call into an MCP-compliant request and sends it to the server (via Stdio or SSE).
- The MCP server executes the operation (e.g., actually reading the file from the disk) and returns the data.

### 4. Feedback (The Result)
The SDK takes the output from the MCP server and appends it to the conversation history as a **`tool_call_result`**.
*   *Internal history update:* `Role: Tool, Content: "Hello World from the file!"`

### 5. Final Synthesis
The `Runner` automatically sends this updated conversation history (User Query + Tool Call + Tool Result) back to the LLM. The LLM now "sees" the content of the file and uses it to generate the final human-readable response.

---

## 8. How It Works (Workflow)

1.  **Initialization:**
    - The script parses CLI arguments to configure the model name, base URL, and which MCP servers to enable.
    - It initializes `MCPServer` objects based on the operating system and user choice.
2.  **Connection:**
    - Before running the query, the script iterates through all configured MCP servers and calls `await server.connect()`.
3.  **Prompting:**
    - The `Agent` is created with instructions and the list of active MCP servers.
    - The `Runner` sends the user query to the LLM (hosted on OVMS).
4.  **Tool Execution:**
    - If the LLM determines it needs a tool (e.g., "List files in /tmp"), it generates a tool call.
    - The `openai-agents` SDK intercepts this, executes the command via the MCP connection, and sends the result back to the LLM.
5.  **Final Response:**
    - The LLM processes the tool output and provides the final answer to the user.

## 5. Usage & CLI Configuration

You can run the agent with various flags to customize its behavior:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--query` | The task for the agent. | `List files in /root` |
| `--model` | The model name hosted on OVMS. | `Qwen/Qwen3-8B` |
| `--base-url` | The OVMS endpoint. | `http://localhost:8000/v3` |
| `--mcp-server` | Choose `weather`, `fs`, or `all`. | `all` |
| `--stream` | Enable real-time token streaming. | `False` |
| `--enable-thinking` | Enable "Thinking" mode for compatible models. | `False` |

**Example Command:**
```bash
python openai_agent.py --query "What is the weather in Paris?" --mcp-server weather --stream
```

## 9. Gemma 4 Support Status & Testing

### The "Gemma 4" Limitation
While `agent.py` is configured to use the `gemma-4-it` model ID, users may encounter an `Unsupported 'gemma4' VLM model type` error when running through OVMS.

**Technical Reasons:**
- **Experimental Architecture:** As of early 2026, Gemma 4 is an experimental model. Support currently exists primarily in a custom Python fork of `optimum-intel` (`support_gemma_4` branch).
- **Backend Discrepancy:** OVMS uses the `openvino-genai` C++ backend for high-performance inference. This backend requires specific architecture mappings for VLMs (Visual Language Models) to handle vision-to-language projection. These mappings have not yet been ported from the experimental Python implementation to the stable C++ production engine.
- **VLM vs LLM:** This model is a VLM, which adds complexity to the inference pipeline compared to standard text-only LLMs.

### Testing with Qwen 2.5
To validate the **Agent SDK logic** and **MCP Tool execution** without being blocked by experimental model support, it is recommended to use **Qwen 2.5 (OpenVINO optimized)** as a fallback.

- **Compatibility:** Qwen 2.5 is fully supported by the OVMS GenAI backend.
- **Workflow:** By serving Qwen 2.5 under the alias `gemma-4-it` and enabling the `--tool_parser hermes3` flag in the Docker configuration, the `agent.py` script can correctly intercept and execute MCP tools.

### Successful Test Verification
The following test was performed to verify the end-to-end integration:
1. **Query:** "Run the bash command 'touch /root/qwen_test_file' on the remote server"
2. **Tool Execution:** The agent successfully called `execute-command` via the SSH MCP server.
3. **Verification:** A direct SSH command confirmed the file's creation:
   ```bash
   -rw-r--r-- 1 root root 0 Apr 28 19:30 /root/qwen_test_file
   ```

---

## 10. Prerequisites


- **Intel OVMS:** An active instance of OpenVINO Model Server running an LLM.
- **Node.js/npx:** Required for the Filesystem MCP server (`@modelcontextprotocol/server-filesystem`).
- **Python Dependencies:** Installed via `pip install -r requirements.txt`.
- **MCP Weather Server:** The `mcp_weather_server` package must be accessible.
