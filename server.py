"""
OpenAI-compatible inference server for Gemma 4 on OpenVINO.
Endpoints:
  GET  /v1/models
  POST /v1/chat/completions  (streaming and non-streaming)
  GET  /health
"""

import time
import uuid
import base64
import io
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

from PIL import Image as PILImage
import uvicorn
import openvino as ov
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoProcessor, TextIteratorStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemma4-server")

# ── Thinking tag constants ────────────────────────────────────────────────────
THINK_START_TAGS  = ["<|channel>thought", "<think>"]
THINK_END_TAGS    = ["<channel|>", "</think>", "<|channel>"]
CLEANUP_TAGS      = ["<turn|>", "<channel|>", "<|channel>", "<end_of_turn>", "<eos>", "<bos>"]
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

# ── EOS tokens to strip when decoding with skip_special_tokens=False ─────────
# These are Gemma-specific tokens that appear as literal strings when
# skip_special_tokens=False but must not appear in the final response.
_EOS_STRIP_TOKENS: list[str] = []

def _build_eos_strip_list() -> list[str]:
    tokens = set()
    # Always strip the known Gemma turn/eos tokens
    tokens.update(["<end_of_turn>", "<eos>", "<bos>"])
    # Also strip whatever the tokenizer reports as eos_token
    try:
        tok = getattr(processor, 'tokenizer', processor)
        if hasattr(tok, 'eos_token') and tok.eos_token:
            tokens.add(tok.eos_token)
        if hasattr(tok, 'additional_special_tokens'):
            for t in tok.additional_special_tokens:
                # Only strip tokens that look like control tokens, not content
                if t.startswith('<') and t.endswith('>') and len(t) < 30:
                    tokens.add(t)
    except Exception as e:
        logger.warning(f"Could not build full EOS strip list: {e}")
    result = sorted(tokens, key=len, reverse=True)  # longest first to avoid partial matches
    logger.info(f"EOS strip tokens: {result}")
    return result

_EOS_STRIP_TOKENS = _build_eos_strip_list()

def _strip_eos_tokens(text: str) -> str:
    """Remove EOS/turn tokens that appear when skip_special_tokens=False."""
    for token in _EOS_STRIP_TOKENS:
        text = text.replace(token, "")
    return text

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

# Identifies this build — update when the image is rebuilt
SYSTEM_FINGERPRINT = f"gemma4-ovms-{MODEL_ID.split('/')[-1]}"
# Fixed registration timestamp for this model build (seconds since epoch)
MODEL_CREATED_AT = 1746748800   # 2025-05-09 00:00:00 UTC

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
    model_config = ConfigDict(extra='ignore')
    role: str
    content: str | list   # str for simple text, list for multimodal

class StreamOptions(BaseModel):
    model_config = ConfigDict(extra='ignore')
    include_usage: Optional[bool] = False

class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra='ignore')
    type: Optional[str] = "text"   # "text" | "json_object" | "json_schema"

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    model:                 str            = MODEL_ID
    messages:              list[Message]
    max_tokens:            Optional[int]  = None
    max_completion_tokens: Optional[int]  = None
    temperature:           Optional[float]= 1.0
    top_p:                 Optional[float]= None
    top_k:                 Optional[int]  = None
    repetition_penalty:    Optional[float]= None
    stop:                  Optional[list[str]] = None
    n:                     Optional[int]  = 1
    stream:                Optional[bool] = False
    stream_options:        Optional[StreamOptions] = None
    do_sample:             Optional[bool] = False
    enable_thinking:       Optional[bool] = False
    reasoning_effort:      Optional[str]  = None     # OpenAI standard: "low", "medium", "high"
    response_format:       Optional[ResponseFormat] = None

# ── Concurrency control ───────────────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = 1   # single model instance
MAX_QUEUED_REQUESTS     = int(os.getenv("MAX_QUEUED_REQUESTS", "4"))
_queue_counter          = 0
_queue_lock             = threading.Lock()

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))
MAX_ACCUMULATE = 512
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT_SECONDS", "600"))

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
    # Return MAX_INPUT_TOKENS so clients (like Hermes) compress their context properly
    # instead of sending 50k tokens which hangs the CPU OpenVINO execution.
    max_ctx = MAX_INPUT_TOKENS
    return {
        "object": "list",
        "data": [{
            "id":             MODEL_ID,
            "object":         "model",
            "created":        MODEL_CREATED_AT,
            "owned_by":       "google/openvino",
            "context_length": max_ctx,
            "max_context_length": max_ctx,     # alias used by some clients
            "system_fingerprint": SYSTEM_FINGERPRINT,
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

def _decode_image_url(url: str) -> PILImage.Image:
    """
    Converts an OpenAI image_url (data URI or http/https URL) to a PIL Image.
    Supports: data:image/<fmt>;base64,<payload>  and  http(s)://<url>
    """
    if url.startswith("data:"):
        # data:image/png;base64,<payload>
        header, encoded = url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        return PILImage.open(io.BytesIO(image_bytes))
    else:
        import requests as _req
        response = _req.get(url, timeout=10)
        response.raise_for_status()
        return PILImage.open(io.BytesIO(response.content))


def _prepare_inputs(messages: list, enable_thinking: bool) -> tuple:
    """
    Converts a list of Message objects into processor inputs.
    Handles OpenAI image_url format by converting to PIL Images.
    Returns (inputs_dict, input_len).
    """
    formatted = []
    for m in messages:
        if isinstance(m.content, list):
            converted_parts = []
            for part in m.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if not url:
                        logger.warning("image_url part has empty url — skipping")
                        continue
                    try:
                        pil_img = _decode_image_url(url)
                        converted_parts.append({"type": "image", "image": pil_img})
                        logger.info(f"Decoded image_url → PIL Image {pil_img.size} {pil_img.mode}")
                    except Exception as e:
                        logger.warning(f"Failed to decode image_url: {e} — skipping image")
                else:
                    converted_parts.append(part)
            content = converted_parts
        else:
            content = [{"type": "text", "text": m.content}]
        formatted.append({"role": m.role, "content": content})

    template_kwargs = {"add_generation_prompt": True, "tokenize": False}
    if THINKING_SUPPORTED and enable_thinking:
        template_kwargs["enable_thinking"] = True

    text = processor.apply_chat_template(formatted, **template_kwargs)

    # Collect PIL images for the processor call
    pil_images = [
        part["image"]
        for m_dict in formatted
        for part in (m_dict["content"] if isinstance(m_dict["content"], list) else [])
        if isinstance(part, dict) and part.get("type") == "image"
    ]

    if pil_images:
        inputs = processor(text=text, images=pil_images, return_tensors="pt")
    else:
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

def _make_chunk(
    request_id: str,
    created: int,
    model_id: str,
    delta: dict,
    finish_reason: str | None = None,
) -> dict:
    return {
        "id":               request_id,
        "object":           "chat.completion.chunk",
        "created":          created,
        "model":            model_id,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }

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
    messages:           list,
    max_new_tokens:     int,
    do_sample:          bool,
    enable_thinking:    bool                = False,
    temperature:        Optional[float]     = None,
    top_p:              Optional[float]     = None,
    top_k:              Optional[int]       = None,
    repetition_penalty: Optional[float]     = None,
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
    raw_response = _strip_eos_tokens(raw_response)
    
    thinking, response = parse_thinking(raw_response)
    response = _clean_tags(response)
    return thinking, response, input_len, output_len, finish_reason


@asynccontextmanager
async def _thread_waiter(gen_thread, bridge_thread):
    try:
        yield
    finally:
        loop = asyncio.get_running_loop()
        if gen_thread.is_alive():
            await loop.run_in_executor(None, gen_thread.join)
        if bridge_thread.is_alive():
            await loop.run_in_executor(None, bridge_thread.join)

async def _stream_response(
    req: ChatCompletionRequest,
    request_id: str,
    created: int,
    effective_do_sample: bool,
    effective_max_tokens: int,
    effective_thinking: bool = False,
):
    """
    Standalone async generator for streaming responses.
    Acquires _inference_semaphore here so it is held for the full duration
    of generation, not just until StreamingResponse is created.
    Decrements _queue_counter on exit via finally.
    """
    global _queue_counter
    try:
        await _inference_semaphore.acquire()
    except asyncio.CancelledError:
        with _queue_lock:
            _queue_counter -= 1
        raise
        
    try:
        thread_started = False
        try:
            loop = asyncio.get_running_loop()
            # Use an asyncio.Queue so tokens can be awaited with a timeout,
            # making the generation watchdog reachable during hangs (Fix 3).
            token_queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _put(text: str | None):
                # Called from the generation thread; bridges to the async loop.
                loop.call_soon_threadsafe(token_queue.put_nowait, text)

            inputs, input_len = _prepare_inputs(req.messages, effective_thinking)

            generate_kwargs = dict(
                **inputs,
                do_sample      = effective_do_sample,
                max_new_tokens = effective_max_tokens,
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

            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,
            )
            generate_kwargs["streamer"] = streamer

            def _generate():
                """Runs model.generate. Tokens are pushed to streamer's internal queue."""
                try:
                    model.generate(**generate_kwargs)
                except Exception as exc:
                    generation_error[0] = exc
                finally:
                    streamer.end()

            def _bridge():
                """
                Reads tokens from the streamer's queue AS THEY ARE GENERATED
                (one blocking read per token) and bridges them to the async token_queue.
                Must run concurrently with _generate — not sequentially.
                """
                try:
                    for token in streamer:   # blocks until each token is ready
                        _put(token)
                finally:
                    _put(None)               # sentinel — always sent, even on error

            gen_thread    = threading.Thread(target=_generate, daemon=True)
            bridge_thread = threading.Thread(target=_bridge,   daemon=True)
            
            def _release_when_done():
                gen_thread.join()
                loop.call_soon_threadsafe(_inference_semaphore.release)
                
            release_thread = threading.Thread(target=_release_when_done, daemon=True)

            gen_thread.start()
            thread_started = True
            bridge_thread.start()
            release_thread.start()

            # OpenAI spec: first chunk always carries role, with empty content
            yield "data: " + json.dumps(
                _make_chunk(request_id, created, req.model, {"role": "assistant", "content": ""})
            ) + "\n\n"

            in_thinking      = False
            accumulated      = ""
            t_start          = time.time()
            generated_tokens = 0
            stop_triggered   = False

            # ── Token consumption loop ────────────────────────────────────────
            wait_start = time.time()
            while True:
                if stop_triggered:
                    break
                try:
                    # Wait for 15s max per loop to yield keep-alives to prevent client disconnects
                    token_text = await asyncio.wait_for(
                        token_queue.get(), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    if time.time() - wait_start > GENERATION_TIMEOUT:
                        logger.error(f"[{request_id}] Generation timed out after {GENERATION_TIMEOUT}s")
                        yield "data: " + json.dumps(
                            _make_chunk(request_id, created, req.model, {}, finish_reason="error")
                        ) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    else:
                        yield ": keep-alive\n\n"
                        continue

                wait_start = time.time() # Reset timeout tracker on token received
                
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
                                    yield "data: " + json.dumps(_make_chunk(request_id, created, req.model, {'content': before})) + "\n\n"
                                in_thinking = True
                                accumulated = accumulated[pos + len(tag):]
                                yield "data: " + json.dumps(_make_chunk(request_id, created, req.model, {'content': '<think>\n'})) + "\n\n"
                                found = True
                                break
                    else:
                        for tag in THINK_END_TAGS:
                            if tag in accumulated:
                                pos    = accumulated.find(tag)
                                before = accumulated[:pos]
                                if before:
                                    yield "data: " + json.dumps(_make_chunk(request_id, created, req.model, {'content': before})) + "\n\n"
                                in_thinking = False
                                accumulated = accumulated[pos + len(tag):]
                                yield "data: " + json.dumps(_make_chunk(request_id, created, req.model, {'content': '\n</think>\n\n'})) + "\n\n"
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
                            yield "data: " + json.dumps(_make_chunk(request_id, created, req.model, {'content': to_yield})) + "\n\n"
                        break   # Exit the token loop — no more tokens needed

                    if to_yield:
                        yield "data: " + json.dumps(_make_chunk(request_id, created, req.model, {'content': to_yield})) + "\n\n"

                await asyncio.sleep(0)
            # ── End of token loop ─────────────────────────────────────────────

            if generation_error[0]:
                logger.error(f"[{request_id}] Generation error: {generation_error[0]}")

            if generated_tokens == 0:
                logger.error(
                    f"[{request_id}] STREAM ENDED with zero generated tokens. "
                    f"generation_error: {generation_error[0]}"
                )

            # Final flush of anything still in the buffer
            if accumulated:
                accumulated = _clean_tags(accumulated)
                if req.stop:
                    accumulated = _apply_stop_sequences(accumulated, req.stop)
                if accumulated:
                    yield "data: " + json.dumps(_make_chunk(request_id, created, req.model, {'content': accumulated})) + "\n\n"

            finish_reason = "length" if generated_tokens >= effective_max_tokens else "stop"
            if generation_error[0]:
                 finish_reason = "error"
            yield "data: " + json.dumps(_make_chunk(request_id, created, req.model, {}, finish_reason)) + "\n\n"

            # Send usage chunk if requested via stream_options
            if req.stream_options and req.stream_options.include_usage:
                usage_chunk = {
                    "id":               request_id,
                    "object":           "chat.completion.chunk",
                    "created":          created,
                    "model":            req.model,
                    "system_fingerprint": SYSTEM_FINGERPRINT,
                    "choices": [],
                    "usage": {
                        "prompt_tokens":     input_len,
                        "completion_tokens": generated_tokens,
                        "total_tokens":      input_len + generated_tokens,
                    },
                    "prompt_tokens": input_len,
                    "completion_tokens": generated_tokens,
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"

            elapsed = time.time() - t_start
            logger.info(
                f"[{request_id}] streamed {generated_tokens} tokens in {elapsed:.2f}s "
                f"({generated_tokens / elapsed:.1f} tok/s)"
            )
        finally:
            if not thread_started:
                _inference_semaphore.release()
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
    effective_max_tokens = req.max_completion_tokens or req.max_tokens or 512

    # Map reasoning_effort → enable_thinking.
    # "high" activates thinking; "low" / "medium" / None leave it off.
    effective_thinking = (
        (req.enable_thinking or False)
        or (req.reasoning_effort is not None and req.reasoning_effort.lower() == "high")
    )

    if req.response_format and req.response_format.type not in (None, "text"):
        logger.warning(
            f"[{request_id}] response_format.type={req.response_format.type!r} "
            "requested but not enforced — returning plain text"
        )

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
            _stream_response(req, request_id, created, effective_do_sample, effective_max_tokens, effective_thinking),
            media_type="text/event-stream"
        )
    else:
        try:
            await _inference_semaphore.acquire()
            loop = asyncio.get_running_loop()
            
            t0 = time.time()
            try:
                thinking, response_text, input_tokens, output_tokens, finish_reason = \
                    await loop.run_in_executor(
                        executor,
                        functools.partial(
                            _run_inference,
                            req.messages,
                            effective_max_tokens,
                            effective_do_sample,
                            effective_thinking,
                            req.temperature,
                            req.top_p,
                            req.top_k,
                            req.repetition_penalty,
                        )
                    )
            finally:
                _inference_semaphore.release()
            
            elapsed = time.time() - t0
            logger.info(f"[{request_id}] Generated {output_tokens} tokens in {elapsed:.2f}s "
                        f"({output_tokens/elapsed:.1f} tok/s)")

            response_text = _apply_stop_sequences(response_text, req.stop)
            
            if thinking:
                response_text = f"<think>\n{thinking}\n</think>\n\n{response_text}"
            
            # Guard: log if content is empty so we can diagnose it
            final_content = response_text.strip()
            if not final_content:
                logger.error(
                    f"[{request_id}] EMPTY CONTENT after processing. "
                    f"raw thinking present: {bool(thinking)}, "
                    f"raw response_text before strip: {response_text!r}, "
                    f"output_tokens: {output_tokens}"
                )

            message = {"role": "assistant", "content": final_content}

            return JSONResponse({
                "id":               request_id,
                "object":           "chat.completion",
                "created":          created,
                "model":            req.model,
                "system_fingerprint": SYSTEM_FINGERPRINT,
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
