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
import functools
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
import openvino as ov
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, TextIteratorStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemma4-server")

# ── Thinking tag constants ────────────────────────────────────────────────────
THINK_START_TAGS  = ["<|channel>thought", "<think>"]
THINK_END_TAGS    = ["<channel|>", "</think>", "<|channel>"]
CLEANUP_TAGS      = ["<turn|>", "<channel|>", "<|channel>"]
# All tags in one flat list — used for prefix-buffer safety in the streamer
ALL_TAGS          = THINK_START_TAGS + THINK_END_TAGS + CLEANUP_TAGS
_MAX_TAG_LEN      = max(len(t) for t in ALL_TAGS)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "/models"
MODEL_ID   = "OpenVINO/gemma-4-E4B-it-int8-ov"

if not os.path.isdir(MODEL_PATH):
    raise RuntimeError(
        f"MODEL_PATH '{MODEL_PATH}' does not exist or is not a directory. "
        "Make sure the model volume is mounted correctly."
    )
required_files = ["config.json"]
for f in required_files:
    if not os.path.exists(os.path.join(MODEL_PATH, f)):
        raise RuntimeError(f"Required model file '{f}' not found in {MODEL_PATH}.")
logger.info(f"Model path validated: {MODEL_PATH}")

logger.info("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

core = ov.Core()
devices = core.available_devices
DEVICE = os.getenv("DEVICE", "GPU" if "GPU" in devices else "CPU")
if DEVICE not in devices and DEVICE != "AUTO":
    logger.warning(
        f"Requested device '{DEVICE}' not found in available devices {devices}. "
        f"Falling back to CPU."
    )
    DEVICE = "CPU"
logger.info(f"Available devices: {devices}")
logger.info(f"Using device: {DEVICE}")

ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

logger.info("Loading model...")
t0 = time.time()
model = OVModelForVisualCausalLM.from_pretrained(MODEL_PATH, device=DEVICE, ov_config=ov_config)
logger.info(f"Model loaded in {time.time() - t0:.1f}s")

try:
    # Report what device the model actually compiled to
    actual_device = getattr(model, '_device', None) or getattr(model.model, 'request', None)
    logger.info(f"Model compiled for device (requested): {DEVICE}")
    logger.info(f"OV_CPU_BACKEND_NUM_THREADS env: {os.getenv('OV_CPU_BACKEND_NUM_THREADS', 'not set')}")
except Exception:
    pass  # best-effort

def _probe_thinking_support() -> bool:
    try:
        test_messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        processor.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        logger.info("Thinking support: ENABLED (enable_thinking kwarg accepted)")
        return True
    except TypeError:
        logger.warning("Thinking support: DISABLED (enable_thinking kwarg rejected by this template)")
        return False

THINKING_SUPPORTED = _probe_thinking_support()

# ── Thread Pool ──────────────────────────────────────────────────────────────
executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

# ── Schema ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str | list   # str for simple text, list for multimodal

class ChatCompletionRequest(BaseModel):
    model:             str            = MODEL_ID
    messages:          list[Message]
    max_tokens:        Optional[int]  = 512
    temperature:       Optional[float]= 1.0
    top_p:             Optional[float]= None
    top_k:             Optional[int]  = None
    repetition_penalty:Optional[float]= None
    stop:              Optional[list[str]] = None
    n:                 Optional[int]  = 1     # only 1 supported; validate below
    stream:            Optional[bool] = False
    do_sample:         Optional[bool] = False
    enable_thinking:   Optional[bool] = False

# ── Concurrency control ───────────────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = 1   # single model instance
MAX_QUEUED_REQUESTS     = int(os.getenv("MAX_QUEUED_REQUESTS", "4"))
_queue_counter          = 0
_queue_lock             = threading.Lock()

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))
MAX_ACCUMULATE = 512
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))

# Semaphore is None until the lifespan initializes it
_inference_semaphore: asyncio.Semaphore | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_semaphore
    _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    logger.info(f"Inference semaphore initialized (max_concurrent={MAX_CONCURRENT_REQUESTS}, max_queued={MAX_QUEUED_REQUESTS})")
    yield
    # Shutdown: nothing to clean up for the semaphore

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Gemma 4 OpenVINO OpenAI API", version="1.0.0", lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

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
    GEMMA4_MAX_CONTEXT = 131072   # Gemma 4 supports 128k context
    try:
        raw = processor.tokenizer.model_max_length
        # Reject placeholder values larger than any real context window
        max_ctx = raw if raw <= GEMMA4_MAX_CONTEXT else GEMMA4_MAX_CONTEXT
    except AttributeError:
        max_ctx = GEMMA4_MAX_CONTEXT
    return {
        "object": "list",
        "data": [{
            "id":             MODEL_ID,
            "object":         "model",
            "created":        int(time.time()),
            "owned_by":       "google/openvino",
            "context_length": max_ctx,
        }]
    }

# ── API Helpers ───────────────────────────────────────────────────────────────
def _error(status: int, message: str, err_type: str = "server_error", code: str | None = None):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": code or str(status)}}
    )

def _apply_stop_sequences(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text
    for s in stop:
        pos = text.find(s)
        if pos != -1:
            text = text[:pos]
    return text

def _clean_tags(text: str) -> str:
    for tag in CLEANUP_TAGS:
        text = text.replace(tag, "")
    return text

def _prepare_inputs(messages: list, enable_thinking: bool) -> tuple:
    """
    Converts a list of Message objects into processor inputs.
    Returns (inputs_dict, input_len).
    """
    formatted = []
    for m in messages:
        content = m.content if isinstance(m.content, list) else [{"type": "text", "text": m.content}]
        formatted.append({"role": m.role, "content": content})

    template_kwargs = {"add_generation_prompt": True, "tokenize": False}
    if THINKING_SUPPORTED and enable_thinking:
        template_kwargs["enable_thinking"] = True

    text   = processor.apply_chat_template(formatted, **template_kwargs)
    inputs = processor(text=text, return_tensors="pt")
    return inputs, inputs["input_ids"].shape[-1]

def _safe_prefix_len(text: str) -> int:
    """
    Returns the length of the longest suffix of `text` that is a
    prefix of any known tag — i.e., the number of characters to hold
    back from flushing because they might be the start of a tag.
    """
    max_hold = 0
    for tag in ALL_TAGS:
        for i in range(1, min(len(text), len(tag)) + 1):
            if tag.startswith(text[-i:]):
                max_hold = max(max_hold, i)
    return max_hold

def parse_thinking(raw_output: str) -> tuple[str, str]:
    """
    Splits model output into (thinking, answer).
    Returns ('', raw_output) if no <think> block is found.
    """
    # Use non-greedy match to extract the *first* think block.
    # We only expect one think block per turn.
    match = re.search(r'<think>(.*?)</think>(.*)', raw_output, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer   = match.group(2).strip()
        return thinking, answer
    
    # Adapt to Gemma 4 specific tags: <|channel>thought ... <channel|> or <|channel>
    if "<|channel>thought" in raw_output:
        start_idx = raw_output.find("<|channel>thought")
        thought_start = start_idx + len("<|channel>thought")
        
        # Search for end tags. <channel|> is unambiguous; <|channel> is also a
        # start-tag prefix so only use it as an end tag if <channel|> is absent.
        end_tags = ["<channel|>", "<|channel>"]   # ordered: unambiguous first
        thought_end    = len(raw_output)
        found_tag_len  = 0

        for tag in end_tags:
            pos = raw_output.find(tag, thought_start)
            if pos != -1 and pos < thought_end:
                thought_end   = pos
                found_tag_len = len(tag)
                break   # stop at first (most unambiguous) match
        
        thinking = raw_output[thought_start:thought_end].strip()
        answer = raw_output[thought_end + found_tag_len:].strip()
        return thinking, answer

    return "", raw_output.strip()

def _run_inference(
    messages: list, max_new_tokens: int, do_sample: bool, enable_thinking: bool = False,
    temperature: float = 1.0, top_p: float = None, top_k: int = None, repetition_penalty: float = None
) -> tuple[str, str, int, int, str]:
    """Returns (thinking, response_text, input_token_count, output_token_count, finish_reason)."""
    inputs, input_len = _prepare_inputs(messages, enable_thinking)

    gen_params = dict(
        do_sample        = do_sample,
        max_new_tokens   = max_new_tokens,
    )
    if temperature and do_sample:
        gen_params["temperature"] = temperature
    if top_p is not None:
        gen_params["top_p"] = top_p
    if top_k is not None:
        gen_params["top_k"] = top_k
    if repetition_penalty is not None:
        gen_params["repetition_penalty"] = repetition_penalty

    output = model.generate(**inputs, **gen_params)
    output_len = output.shape[-1] - input_len
    finish_reason = "length" if output_len >= max_new_tokens else "stop"
    
    # We use skip_special_tokens=False to catch the thinking tags
    raw_response = processor.decode(output[0][input_len:], skip_special_tokens=False)
    
    thinking, response = parse_thinking(raw_response)
    response = _clean_tags(response)
    return thinking, response, input_len, output_len, finish_reason


async def _stream_response(req: ChatCompletionRequest, request_id: str, created: int, effective_do_sample: bool):
    """
    Standalone async generator for streaming responses.
    Acquires _inference_semaphore here so it is held for the full duration
    of generation, not just until StreamingResponse is created.
    Decrements _queue_counter on exit via finally.
    """
    global _queue_counter
    try:
        async with _inference_semaphore:
            loop = asyncio.get_running_loop()
            # Use an asyncio.Queue so tokens can be awaited with a timeout,
            # making the generation watchdog reachable during hangs (Fix 3).
            token_queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _put(text: str | None):
                # Called from the generation thread; bridges to the async loop.
                loop.call_soon_threadsafe(token_queue.put_nowait, text)

            inputs, input_len = _prepare_inputs(req.messages, req.enable_thinking or False)

            generate_kwargs = dict(
                **inputs,
                do_sample      = effective_do_sample,
                max_new_tokens = req.max_tokens or 512,
            )
            if req.temperature and effective_do_sample:
                generate_kwargs["temperature"] = req.temperature
            if req.top_p is not None:
                generate_kwargs["top_p"] = req.top_p
            if req.top_k is not None:
                generate_kwargs["top_k"] = req.top_k
            if req.repetition_penalty is not None:
                generate_kwargs["repetition_penalty"] = req.repetition_penalty
            if req.stop:
                logger.warning(f"[{request_id}] stop sequences: post-hoc truncation only")

            generation_error: list[BaseException | None] = [None]

            def _generate():
                """Runs model.generate in a thread, pushes tokens via _put."""
                streamer = TextIteratorStreamer(
                    processor.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=False,
                )
                generate_kwargs["streamer"] = streamer
                try:
                    model.generate(**generate_kwargs)
                except Exception as exc:
                    generation_error[0] = exc
                finally:
                    # Always push the sentinel so the consumer loop can exit.
                    for token in streamer:
                        _put(token)
                    _put(None)  # sentinel

            thread = threading.Thread(target=_generate, daemon=True)
            thread.start()

            in_thinking      = False
            accumulated      = ""
            t_start          = time.time()
            generated_tokens = 0
            stop_triggered   = False

            def make_chunk(delta_dict, finish_reason=None):
                return {
                    "id":      request_id,
                    "object":  "chat.completion.chunk",
                    "created": created,
                    "model":   req.model,
                    "choices": [{"index": 0, "delta": delta_dict, "finish_reason": finish_reason}],
                }

            # ── Token consumption loop ────────────────────────────────────────
            while True:
                if stop_triggered:
                    break
                try:
                    # asyncio.wait_for makes the timeout reachable even when
                    # model.generate hangs and never produces a token (Fix 3).
                    token_text = await asyncio.wait_for(
                        token_queue.get(), timeout=GENERATION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[{request_id}] Generation timed out after {GENERATION_TIMEOUT}s")
                    yield f"data: {json.dumps({'error': 'generation timeout'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if token_text is None:   # sentinel — generation finished
                    break

                generated_tokens += 1
                accumulated += token_text

                # ── Tag-transition state machine ──────────────────────────────
                while True:
                    found = False
                    if not in_thinking:
                        for tag in THINK_START_TAGS:
                            if tag in accumulated:
                                pos    = accumulated.find(tag)
                                before = _clean_tags(accumulated[:pos])
                                if before:
                                    yield f"data: {json.dumps(make_chunk({'content': before}))}\n\n"
                                in_thinking = True
                                accumulated = accumulated[pos + len(tag):]
                                yield "data: " + json.dumps(make_chunk({'content': '<think>\n'})) + "\n\n"
                                found = True
                                break
                    else:
                        for tag in THINK_END_TAGS:
                            if tag in accumulated:
                                pos    = accumulated.find(tag)
                                before = accumulated[:pos]
                                if before:
                                    yield f"data: {json.dumps(make_chunk({'content': before}))}\n\n"
                                in_thinking = False
                                accumulated = accumulated[pos + len(tag):]
                                yield "data: " + json.dumps(make_chunk({'content': '\n</think>\n\n'})) + "\n\n"
                                found = True
                                break
                    if not found:
                        break

                # ── Safe-flush: hold back any tag prefix ──────────────────────
                tail_len = _safe_prefix_len(accumulated)
                if len(accumulated) > MAX_ACCUMULATE:
                    tail_len = max(tail_len, _MAX_TAG_LEN)

                if len(accumulated) > tail_len:
                    to_yield    = accumulated[:-tail_len] if tail_len > 0 else accumulated
                    accumulated = accumulated[-tail_len:] if tail_len > 0 else ""
                    to_yield    = _clean_tags(to_yield)
                    
                    if req.stop and not stop_triggered:
                        truncated = _apply_stop_sequences(to_yield, req.stop)
                        if len(truncated) < len(to_yield):
                            stop_triggered = True
                            to_yield = truncated
                        else:
                            to_yield = truncated
                            
                    if stop_triggered:
                        if to_yield:
                            yield f"data: {json.dumps(make_chunk({'content': to_yield}))}\n\n"
                        break   # Exit the token loop — no more tokens needed

                    if to_yield:
                        yield f"data: {json.dumps(make_chunk({'content': to_yield}))}\n\n"

                await asyncio.sleep(0)
            # ── End of token loop ─────────────────────────────────────────────

            await loop.run_in_executor(None, thread.join)

            if generation_error[0]:
                logger.error(f"[{request_id}] Generation error: {generation_error[0]}")

            # Final flush of anything still in the buffer
            if accumulated:
                accumulated = _clean_tags(accumulated)
                if req.stop:
                    accumulated = _apply_stop_sequences(accumulated, req.stop)
                if accumulated:
                    yield f"data: {json.dumps(make_chunk({'content': accumulated}))}\n\n"

            finish_reason = "length" if generated_tokens >= (req.max_tokens or 512) else "stop"
            yield f"data: {json.dumps(make_chunk({}, finish_reason))}\n\n"
            yield "data: [DONE]\n\n"

            elapsed = time.time() - t_start
            logger.info(
                f"[{request_id}] streamed {generated_tokens} tokens in {elapsed:.2f}s "
                f"({generated_tokens / elapsed:.1f} tok/s)"
            )

    finally:
        with _queue_lock:
            _queue_counter -= 1

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created    = int(time.time())

    logger.info(f"[{request_id}] stream={req.stream} thinking={req.enable_thinking} max_tokens={req.max_tokens}")

    if req.n and req.n != 1:
        return _error(400, "Only n=1 is supported.", "invalid_request_error", "unsupported_n")

    total_chars = sum(
        len(m.content) if isinstance(m.content, str) else sum(
            len(p.get("text", "")) for p in m.content if isinstance(p, dict)
        )
        for m in req.messages
    )
    if total_chars > MAX_INPUT_TOKENS * 6:
        return _error(400, f"Request too large (estimated input exceeds {MAX_INPUT_TOKENS} tokens).",
                      "invalid_request_error", "context_length_exceeded")

    effective_do_sample = req.do_sample or (req.temperature is not None and req.temperature != 1.0)

    # ── Queue capacity check ──────────────────────────────────────────────────
    global _queue_counter
    with _queue_lock:
        if _queue_counter >= MAX_QUEUED_REQUESTS:
            return _error(503, "Server at capacity, try again later.", "server_error", "503")
        _queue_counter += 1
    # NOTE: _queue_counter is decremented inside the generator (streaming)
    # or in the finally block below (non-streaming).

    if req.stream:
        return StreamingResponse(
            _stream_response(req, request_id, created, effective_do_sample),
            media_type="text/event-stream"
        )
    else:
        try:
            async with _inference_semaphore:
                loop = asyncio.get_running_loop()
                t0 = time.time()
                thinking, response_text, input_tokens, output_tokens, finish_reason = \
                    await loop.run_in_executor(
                        executor,
                        functools.partial(
                            _run_inference,
                            req.messages,
                            req.max_tokens or 512,
                            effective_do_sample,
                            req.enable_thinking or False,
                            req.temperature,
                            req.top_p,
                            req.top_k,
                            req.repetition_penalty,
                        )
                    )
            elapsed = time.time() - t0
            logger.info(f"[{request_id}] Generated {output_tokens} tokens in {elapsed:.2f}s "
                        f"({output_tokens/elapsed:.1f} tok/s)")

            response_text = _apply_stop_sequences(response_text, req.stop)
            
            if thinking:
                response_text = f"<think>\n{thinking}\n</think>\n\n{response_text}"
            
            message = {"role": "assistant", "content": response_text.strip()}

            return JSONResponse({
                "id":      request_id,
                "object":  "chat.completion",
                "created": created,
                "model":   req.model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens":     input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens":      input_tokens + output_tokens,
                }
            })
        finally:
            with _queue_lock:
                _queue_counter -= 1

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
