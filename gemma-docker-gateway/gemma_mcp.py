import asyncio
import sys
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Optimum / Transformers / Pillow imports for Gemma 4
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
from PIL import Image

async def main():
    print("Loading Gemma 4 Model with OpenVINO...", file=sys.stderr)
    model_id = "OpenVINO/gemma-4-E4B-it-int8-ov"
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)
        print("Model loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Connecting to MCP Gateway...", file=sys.stderr)
    server_params = StdioServerParameters(
        command="docker",
        args=["mcp", "gateway", "run", "--profile", "assistant"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools_response = await session.list_tools()
            print("Available tools:", file=sys.stderr)
            for tool in tools_response.tools:
                print(f" - {tool.name}: {tool.description}", file=sys.stderr)
            
            search_query = "What is OpenVINO Intel Optimum?"
            print(f"Calling search tool with query: {search_query}...", file=sys.stderr)
            
            tool_found = None
            for tool in tools_response.tools:
                if "search" in tool.name.lower():
                    tool_found = tool.name
                    break
            
            search_result_text = ""
            if tool_found:
                print(f"Using tool {tool_found}...", file=sys.stderr)
                try:
                    result = await session.call_tool(
                        tool_found,
                        arguments={"query": search_query}
                    )
                    
                    for content in result.content:
                        if content.type == "text":
                            search_result_text += content.text + "\n"
                    print(f"Search Result snippet: {search_result_text[:200]}...", file=sys.stderr)
                except Exception as e:
                    print(f"Tool call failed: {e}", file=sys.stderr)
                    search_result_text = "Error calling search."
            else:
                print("No search tool found.", file=sys.stderr)
                search_result_text = "No search info available."
            
            print("Generating response with Gemma 4...", file=sys.stderr)
            img = Image.new('RGB', (224, 224), color='white')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Based on the following search result:\n{search_result_text[:500]}\n\nCan you summarize what OpenVINO Intel Optimum is? Keep it brief."},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[img], return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]
            
            output = model.generate(**inputs, do_sample=False, max_new_tokens=150)
            response = processor.decode(output[0][input_len:], skip_special_tokens=True)
            print("\n--- Gemma 4 Response ---")
            print(response)

if __name__ == "__main__":
    asyncio.run(main())
