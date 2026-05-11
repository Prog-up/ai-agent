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
import concurrent.futures
from typing import Optional, List, AsyncIterator, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemma4-server")

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "/models"
MODEL_ID   = "OpenVINO/gemma-4-E4B-it-int8-ov"

logger.info("Loading processor...")
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

processor = AutoProcessor.from_pretrained(MODEL_PATH)

logger.info("Loading model...")
t0 = time.time()
model = OVModelForVisualCausalLM.from_pretrained(MODEL_PATH)
logger.info(f"Model loaded in {time.time() - t0:.1f}s")

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

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Gemma 4 OpenVINO OpenAI API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

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

def _run_inference(messages: list, max_new_tokens: int, do_sample: bool) -> tuple[str, int, int]:
    """Returns (response_text, input_token_count, output_token_count)."""
    # Convert messages to the format expected by the processor
    formatted = []
    for m in messages:
        content = m.content if isinstance(m.content, list) else [{"type": "text", "text": m.content}]
        formatted.append({"role": m.role, "content": content})

    text = processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]

    output = model.generate(**inputs, do_sample=do_sample, max_new_tokens=max_new_tokens)
    output_len = output.shape[-1] - input_len
    response = processor.decode(output[0][input_len:], skip_special_tokens=True)
    return response, input_len, output_len

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        async def stream_response() -> AsyncIterator[str]:
            # Run inference in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response_text, input_tokens, output_tokens = await loop.run_in_executor(
                executor, _run_inference, req.messages, req.max_tokens or 512, req.do_sample or False
            )
            # Stream word by word
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk_text = word + (" " if i < len(words) - 1 else "")
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": chunk_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)  # yield to event loop

            # Final chunk
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    else:
        loop = asyncio.get_event_loop()
        t0 = time.time()
        response_text, input_tokens, output_tokens = await loop.run_in_executor(
            executor, _run_inference, req.messages, req.max_tokens or 512, req.do_sample or False
        )
        elapsed = time.time() - t0
        logger.info(f"Generated {output_tokens} tokens in {elapsed:.2f}s ({output_tokens/elapsed:.1f} tok/s)")

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
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
