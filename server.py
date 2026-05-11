"""
OpenAI-compatible inference server for Gemma 4 on OpenVINO.
Endpoints:
  GET  /v1/models
  POST /v1/chat/completions  (streaming and non-streaming)
  GET  /health
"""

import time
import uuid
import json
import asyncio
import logging
import os
import re
import threading
import concurrent.futures
from typing import Optional, List, AsyncIterator, Union

import uvicorn
import openvino as ov
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, TextIteratorStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemma4-server")

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "/models"
MODEL_ID   = "OpenVINO/gemma-4-E4B-it-int8-ov"

logger.info("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

core = ov.Core()
devices = core.available_devices
DEVICE = os.getenv("DEVICE", "GPU" if "GPU" in devices else "CPU")
logger.info(f"Available devices: {devices}")
logger.info(f"Using device: {DEVICE}")

ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

logger.info("Loading model...")
t0 = time.time()
model = OVModelForVisualCausalLM.from_pretrained(MODEL_PATH, device=DEVICE, ov_config=ov_config)
logger.info(f"Model loaded in {time.time() - t0:.1f}s")

# Check thinking support
# Gemma 4 supports enable_thinking via **kwargs in apply_chat_template
THINKING_SUPPORTED = True

# ── Thread Pool ──────────────────────────────────────────────────────────────
executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

# ── Schema ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: Union[str, List]   # str for simple text, list for multimodal

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False
    do_sample: Optional[bool] = False
    enable_thinking: Optional[bool] = False

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Gemma 4 OpenVINO OpenAI API", version="1.0.0")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "available_devices": list(core.available_devices),
        "thinking_supported": THINKING_SUPPORTED
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google/openvino",
        }]
    }

def parse_thinking(raw_output: str) -> tuple[str, str]:
    """
    Splits model output into (thinking, answer).
    Returns ('', raw_output) if no <think> block is found.
    """
    # Try the requested <think> tags first
    match = re.search(r'<think>(.*?)</think>(.*)', raw_output, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer   = match.group(2).strip()
        return thinking, answer
    
    # Adapt to Gemma 4 specific tags: <|channel>thought ... <channel|> or <|channel>
    if "<|channel>thought" in raw_output:
        start_idx = raw_output.find("<|channel>thought")
        thought_start = start_idx + len("<|channel>thought")
        
        # Look for end tags
        end_tags = ["<channel|>", "<|channel>"]
        thought_end = len(raw_output)
        found_tag_len = 0
        
        for tag in end_tags:
            pos = raw_output.find(tag, thought_start)
            if pos != -1 and pos < thought_end:
                thought_end = pos
                found_tag_len = len(tag)
        
        thinking = raw_output[thought_start:thought_end].strip()
        answer = raw_output[thought_end + found_tag_len:].strip()
        return thinking, answer

    return "", raw_output.strip()

def _run_inference(messages: list, max_new_tokens: int, do_sample: bool, enable_thinking: bool = False) -> tuple[str, str, int, int]:
    """Returns (thinking, response_text, input_token_count, output_token_count)."""
    # Convert messages to the format expected by the processor
    formatted = []
    for m in messages:
        content = m.content if isinstance(m.content, list) else [{"type": "text", "text": m.content}]
        formatted.append({"role": m.role, "content": content})

    template_kwargs = {"add_generation_prompt": True, "tokenize": False}
    if THINKING_SUPPORTED and enable_thinking:
        template_kwargs["enable_thinking"] = True

    text = processor.apply_chat_template(formatted, **template_kwargs)
    inputs = processor(text=text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]

    output = model.generate(**inputs, do_sample=do_sample, max_new_tokens=max_new_tokens)
    output_len = output.shape[-1] - input_len
    
    # We use skip_special_tokens=False to catch the thinking tags
    raw_response = processor.decode(output[0][input_len:], skip_special_tokens=False)
    
    thinking, response = parse_thinking(raw_response)
    # Clean up any remaining tags in the final response
    response = response.replace("<turn|>", "").replace("<channel|>", "").replace("<|channel>", "").strip()
    return thinking, response, input_len, output_len

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        async def stream_response() -> AsyncIterator[str]:
            # Run inference in a thread with TextIteratorStreamer
            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=False
            )

            template_kwargs = {"add_generation_prompt": True, "tokenize": False}
            if THINKING_SUPPORTED and req.enable_thinking:
                template_kwargs["enable_thinking"] = True

            formatted = []
            for m in req.messages:
                content = m.content if isinstance(m.content, list) else [{"type": "text", "text": m.content}]
                formatted.append({"role": m.role, "content": content})

            text = processor.apply_chat_template(formatted, **template_kwargs)
            inputs = processor(text=text, return_tensors="pt")

            generate_kwargs = dict(
                **inputs,
                streamer=streamer,
                do_sample=req.do_sample or False,
                max_new_tokens=req.max_tokens or 512
            )

            # Start generation in background thread
            thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
            thread.start()

            in_thinking = False
            accumulated = ""
            # Tags that we need to detect and could be split across tokens
            tags = ["<|channel>thought", "<think>", "<channel|>", "</think>", "<|channel>", "<turn|>"]

            def make_chunk(delta_dict):
                return {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": delta_dict, "finish_reason": None}]
                }

            for token_text in streamer:
                accumulated += token_text
                
                while True:
                    found_transition = False
                    if not in_thinking:
                        # Look for start tags
                        for tag in ["<|channel>thought", "<think>"]:
                            if tag in accumulated:
                                pos = accumulated.find(tag)
                                before = accumulated[:pos].replace("<turn|>", "").replace("<channel|>", "").replace("<|channel>", "")
                                if before:
                                    yield f"data: {json.dumps(make_chunk({'content': before}))}\n\n"
                                in_thinking = True
                                accumulated = accumulated[pos + len(tag):]
                                found_transition = True
                                break
                    else:
                        # Look for end tags
                        for tag in ["<channel|>", "</think>", "<|channel>"]:
                            if tag in accumulated:
                                pos = accumulated.find(tag)
                                before = accumulated[:pos]
                                if before:
                                    yield f"data: {json.dumps(make_chunk({'thinking': before}))}\n\n"
                                in_thinking = False
                                accumulated = accumulated[pos + len(tag):]
                                found_transition = True
                                break
                    if not found_transition:
                        break
                
                # Yield what's stable (not a prefix of any tag)
                tail_len = 0
                for tag in tags:
                    for i in range(1, min(len(accumulated), len(tag)) + 1):
                        if tag.startswith(accumulated[-i:]):
                            tail_len = max(tail_len, i)
                
                if len(accumulated) > tail_len:
                    to_yield = accumulated[:-tail_len] if tail_len > 0 else accumulated
                    accumulated = accumulated[-tail_len:] if tail_len > 0 else ""
                    if to_yield:
                        to_yield = to_yield.replace("<turn|>", "").replace("<channel|>", "").replace("<|channel>", "")
                        if to_yield:
                            delta = {"thinking": to_yield} if in_thinking else {"content": to_yield}
                            yield f"data: {json.dumps(make_chunk(delta))}\n\n"
                
                await asyncio.sleep(0)

            thread.join()

            # Final flush
            if accumulated:
                accumulated = accumulated.replace("<turn|>", "").replace("<channel|>", "").replace("<|channel>", "").strip()
                if accumulated:
                    delta = {"thinking": accumulated} if in_thinking else {"content": accumulated}
                    yield f"data: {json.dumps(make_chunk(delta))}\n\n"

            # Final done chunk
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    else:
        loop = asyncio.get_event_loop()
        t0 = time.time()
        thinking, response_text, input_tokens, output_tokens = await loop.run_in_executor(
            executor, _run_inference, req.messages, req.max_tokens or 512, req.do_sample or False, req.enable_thinking or False
        )
        elapsed = time.time() - t0
        logger.info(f"Generated {output_tokens} tokens in {elapsed:.2f}s ({output_tokens/elapsed:.1f} tok/s)")

        message = {"role": "assistant", "content": response_text}
        if thinking:
            message["thinking"] = thinking

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
