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
    Agent, Model, ModelProvider, OpenAIChatCompletionsModel, RunConfig, Runner,
)

API_KEY = "not_used"

async def run_query(query, agent, OVMS_MODEL_PROVIDER, stream: bool = False):
    for server in agent.mcp_servers:
        await server.connect()
    
    if stream:
        print("Gemma: ", end="", flush=True)
        result = Runner.run_streamed(
            starting_agent=agent, input=query, 
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
            starting_agent=agent, input=query, 
            run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER, tracing_disabled=True)
        )
        print(f"Gemma: {result.final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Agent with SSH MCP server.")
    parser.add_argument("--query", type=str, help="Query to run")
    parser.add_argument("--model", type=str, default="gemma-4-it", help="Model name to use")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v3", help="Base URL for the OpenAI API")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--tool-choice", type=str, default="auto", choices=["auto", "required"])
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    ssh_server = MCPServerStdio(
        name="SSH MCP Server",
        params={
            "command": "npx",
            "args": [
                "-y", "@fangjunjie/ssh-mcp-server",
                "--host", "192.168.137.122", "--port", "22",
                "--username", "root", "--privateKey", "/root/.ssh/id_ed25519"
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
        name="Assistant", mcp_servers=[ssh_server],
        model_settings=ModelSettings(
            tool_choice=args.tool_choice, temperature=0.0, max_tokens=1000, 
            extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}
        ),
    )

    if args.query:
        asyncio.run(run_query(args.query, agent, OVMS_MODEL_PROVIDER, args.stream))
