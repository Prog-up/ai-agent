import asyncio
import os
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer
from PIL import Image
import requests
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

# MCP Server Parameters from main.py
server_params = StdioServerParameters(
    command="npx",
    args=["-y",
        "@fangjunjie/ssh-mcp-server",
        "--host", "10.52.171.93",
        "--port", "22",
        "--username", "root",
        "--privateKey", "./ssh/id_ed25519"],
    env={"UV_INDEX": os.environ.get("UV_INDEX", "")},
)

async def chat_loop(model, processor, session):
    print("\n" + "="*30)
    print("      GEMMA-4 MCP CHATBOT")
    print("="*30)
    
    # List available tools to the user
    tools_resp = await session.list_tools()
    tools = tools_resp.tools
    tool_names = [t.name for t in tools]
    print(f"\nConnected to SSH MCP Server. Available tools: {tool_names}")
    
    messages = []
    print("\nChat started! Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            # Use loop.run_in_executor for synchronous input in async loop
            user_text = await asyncio.get_event_loop().run_in_executor(None, input, "\nYou: ")
            user_text = user_text.strip()
            
            if not user_text:
                continue
            if user_text.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Simple manual tool trigger
            if user_text.startswith("/run "):
                cmd = user_text[5:]
                print(f"Executing SSH command: {cmd}...")
                result = await session.call_tool("run_command", arguments={"command": cmd})
                print(f"Tool result: {result.content[0].text}")
                continue

            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            })

            # Prepare inputs (Text-only)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]

            print("Gemma: ", end="", flush=True)
            streamer = TextStreamer(processor, skip_prompt=True, skip_special_tokens=True)
            
            output = model.generate(**inputs, streamer=streamer, do_sample=False, max_new_tokens=512)
            
            # Extract only the newly generated part for the conversation history
            response = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
            
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            })
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

async def main():
    print("Loading model to GPU (this may take a moment)...")
    model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load model with fixed f32 precision for GPU
    ov_config = {"INFERENCE_PRECISION_HINT": "f32"}
    model = OVModelForVisualCausalLM.from_pretrained(model_id, device="GPU", ov_config=ov_config)

    print("Connecting to MCP server...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await chat_loop(model, processor, session)
    except Exception as e:
        print(f"Failed to connect to MCP server: {e}")

if __name__ == "__main__":
    asyncio.run(main())
