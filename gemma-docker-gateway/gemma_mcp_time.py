import asyncio
import sys
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
            
            # 1. Test get_current_time
            timezone = "Asia/Tokyo"
            print(f"Calling time tool for timezone: {timezone}...", file=sys.stderr)
            time_result_text = ""
            try:
                time_result = await session.call_tool(
                    "get_current_time",
                    arguments={"timezone": timezone}
                )
                for content in time_result.content:
                    if content.type == "text":
                        time_result_text += content.text + "\n"
                print(f"Time Result: {time_result_text.strip()}", file=sys.stderr)
            except Exception as e:
                print(f"Time tool call failed: {e}", file=sys.stderr)
                time_result_text = "Error calling time tool."
            
            # 2. Test sequentialthinking
            print("Calling sequentialthinking tool...", file=sys.stderr)
            thought_result_text = ""
            try:
                thought_result = await session.call_tool(
                    "sequentialthinking",
                    arguments={
                        "thought": "I need to analyze why OpenVINO is good for inference on edge devices.",
                        "thoughtNumber": 1,
                        "totalThoughts": 2,
                        "nextThoughtNeeded": True
                    }
                )
                for content in thought_result.content:
                    if content.type == "text":
                        thought_result_text += content.text + "\n"
                print(f"Thought Result: {thought_result_text.strip()}", file=sys.stderr)
            except Exception as e:
                print(f"Sequentialthinking call failed: {e}", file=sys.stderr)
                thought_result_text = "Error calling sequential thinking."

            print("Generating response with Gemma 4...", file=sys.stderr)
            img = Image.new('RGB', (224, 224), color='white')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"A user wants to know the time in Tokyo and why OpenVINO is good for edge devices. Our MCP tools returned the following contexts:\nTime: '{time_result_text.strip()}'\nThought process: '{thought_result_text.strip()}'\n\nPlease provide a brief response incorporating this information."},
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
