#
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import asyncio
import os
import argparse

from openai import AsyncOpenAI
from agents import Agent, Runner, RunConfig
from agents.mcp import MCPServerStdio
from agents.model_settings import ModelSettings
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
)

API_KEY = "not_used"

async def chat_loop(agent, OVMS_MODEL_PROVIDER, stream: bool = False):
    print("\n" + "="*30)
    print("      OVMS MCP SSH CHATBOT")
    print("="*30)
    
    # Connect to MCP servers
    for server in agent.mcp_servers:
        await server.connect()
        print(f"\nConnected to {server.name}.")
    
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

            if stream:
                print("Gemma: ", end="", flush=True)
                result = Runner.run_streamed(
                    starting_agent=agent, 
                    input=user_text, 
                    run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER, tracing_disabled=True)
                )

                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        print(event.data.delta, end="", flush=True)
                    elif event.type == "run_item_stream_event":
                        if event.item.type == "tool_call_item":
                            print(f"\n-- Tool '{event.item.raw_item.name}' called with: {event.item.raw_item.arguments}")
                        elif event.item.type == "tool_call_output_item":
                            print(f"\n-- Tool output received.\n")
                print() # New line after stream
            else:
                result = await Runner.run(
                    starting_agent=agent, 
                    input=user_text, 
                    run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER, tracing_disabled=True)
                )
                print(f"Gemma: {result.final_output}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

async def run_query(query, agent, OVMS_MODEL_PROVIDER, stream: bool = False):
    # Connect to MCP servers
    for server in agent.mcp_servers:
        await server.connect()
    
    if stream:
        print("Gemma: ", end="", flush=True)
        result = Runner.run_streamed(
            starting_agent=agent, 
            input=query, 
            run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER, tracing_disabled=True)
        )

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print(f"\n-- Tool '{event.item.raw_item.name}' called with: {event.item.raw_item.arguments}")
                elif event.item.type == "tool_call_output_item":
                    print(f"\n-- Tool output received.\n")
        print() 
    else:
        result = await Runner.run(
            starting_agent=agent, 
            input=query, 
            run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER, tracing_disabled=True)
        )
        print(f"Gemma: {result.final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Agent with SSH MCP server.")
    parser.add_argument("--query", type=str, help="Query to run (non-interactive mode)")
    parser.add_argument("--model", type=str, default="gemma-4-it", help="Model name to use")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v3", help="Base URL for the OpenAI API")
    parser.add_argument("--stream", action="store_true", help="Stream output from the agent")
    parser.add_argument("--tool-choice", type=str, default="auto", choices=["auto", "required"], help="Tool choice for the agent")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable agent thinking")
    
    args = parser.parse_args()

    # SSH MCP Server configuration from gemma4.py
    ssh_server = MCPServerStdio(
        name="SSH MCP Server",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@fangjunjie/ssh-mcp-server",
                "--host", "10.52.171.93",
                "--port", "22",
                "--username", "root",
                "--privateKey", "./ssh/id_ed25519"
            ],
            "env": {"UV_INDEX": os.environ.get("UV_INDEX", "")}
        }
    )

    client = AsyncOpenAI(base_url=args.base_url, api_key=API_KEY)

    class OVMSModelProvider(ModelProvider):
        def get_model(self, _) -> Model:
            return OpenAIChatCompletionsModel(model=args.model, openai_client=client)

    OVMS_MODEL_PROVIDER = OVMSModelProvider()

    agent = Agent(
        name="Assistant",
        mcp_servers=[ssh_server],
        model_settings=ModelSettings(
            tool_choice=args.tool_choice, 
            temperature=0.0, 
            max_tokens=1000, 
            extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}
        ),
    )

    if args.query:
        asyncio.run(run_query(args.query, agent, OVMS_MODEL_PROVIDER, args.stream))
    else:
        asyncio.run(chat_loop(agent, OVMS_MODEL_PROVIDER, args.stream))
