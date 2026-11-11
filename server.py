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
from typing import Optional
from contextlib import asynccontextmanager

from PIL import Image as PILImage
import uvicorn
import openvino as ov
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoProcessor, TextIteratorStreamer, TextStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("gemma4-server")

# ── Thinking tag constants ────────────────────────────────────────────────────
THINK_START_TAGS  = ["<|channel>thought", "<think>"]
THINK_END_TAGS    = ["<channel|>", "</think>", "<|channel>"]
CLEANUP_TAGS      = ["<turn|>", "<channel|>", "<|channel>", "<end_of_turn>", "<eos>", "<bos>"]
# Tags that mark thinking boundaries — must NEVER be stripped from raw output
# before parse_thinking runs, even if the tokenizer registers them as special tokens.
_THINKING_BOUNDARY_TAGS: frozenset[str] = frozenset(
    THINK_START_TAGS + THINK_END_TAGS
)
# All tags in one flat list — used for prefix-buffer safety in the streamer
ALL_TAGS          = THINK_START_TAGS + THINK_END_TAGS + CLEANUP_TAGS
_MAX_TAG_LEN      = max(len(t) for t in ALL_TAGS)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "/models"
MODEL_ID   = os.getenv("MODEL_ID", "OpenVINO/gemma-4-E4B-it-int8-ov")

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
                    # Never strip thinking boundary markers — parse_thinking needs them
                    if t in _THINKING_BOUNDARY_TAGS:
                        logger.debug(f"Skipping thinking boundary tag from EOS strip list: {t!r}")
                        continue
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

core    = None
devices = []
DEVICE  = "GPU"
ov_config = {}
model   = None
available_devices_list = []

# ── Persistent cache ──────────────────────────────────────────────────────────
OPENVINO_CACHE_DIR = os.getenv("OPENVINO_CACHE_DIR", "/tmp/model_cache")
os.makedirs(OPENVINO_CACHE_DIR, exist_ok=True)
logger.info(f"OpenVINO model cache: {OPENVINO_CACHE_DIR}")

def _initialize_model():
    global core, devices, DEVICE, ov_config, model, available_devices_list
    logger.info("Initializing OpenVINO runtime and loading model on inference thread...")
    t0 = time.time()
    
    core = ov.Core()
    devices = list(core.available_devices)
    available_devices_list = devices
    DEVICE = os.getenv("DEVICE", "GPU" if "GPU" in devices else "CPU")
    
    if DEVICE not in devices and DEVICE != "AUTO":
        logger.warning(
            f"Requested device '{DEVICE}' not found in {devices}. Falling back to CPU."
        )
        DEVICE = "CPU"
        
    logger.info(f"Available devices: {devices}")
    logger.info(f"Using device: {DEVICE}")
    
    # ── Device-conditional ov_config ──────────────────────────────────────────────
    if "GPU" in DEVICE:
        # Intel iGPU / Arc GPU: optimize for single-request latency.
        # KV_CACHE_PRECISION u8 keeps the KV cache in INT8 on GPU memory,
        # halving KV cache VRAM vs f32 — critical for iGPU where model and
        # KV cache compete for shared system RAM.
        # DYNAMIC_QUANTIZATION_GROUP_SIZE enables runtime INT8 quantization
        # of activations in groups of 32, leveraging Intel XMX units on
        # 12th-gen+ Iris Xe / Arc. Set to 0 to disable.
        ov_config = {
            "PERFORMANCE_HINT":              "LATENCY",
            "INFERENCE_PRECISION_HINT":      os.getenv("OV_INFERENCE_PRECISION", "f32"),
            "KV_CACHE_PRECISION":            os.getenv("OV_KV_CACHE_PRECISION", "f16"),
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": os.getenv("OV_DYN_QUANT_GROUP_SIZE", "0"),
            "CACHE_DIR":                     OPENVINO_CACHE_DIR,
        }
        logger.info(f"GPU ov_config: {ov_config}")
    else:
        # Intel CPU: optimize for single-request latency.
        # NUM_STREAMS 1 avoids parallel stream overhead for sequential generation.
        # AFFINITY NUMA binds threads to a single NUMA node — critical for
        # dual-socket systems (prevents cross-socket memory access penalty).
        # f32 precision keeps output quality on INT4/INT8 quantized weights.
        ov_config = {
            "PERFORMANCE_HINT":      "LATENCY",
            "INFERENCE_PRECISION_HINT": os.getenv("OV_INFERENCE_PRECISION", "f32"),
            "NUM_STREAMS":           "1",
            "AFFINITY":              os.getenv("OV_AFFINITY", "NUMA"),
            "CACHE_DIR":             OPENVINO_CACHE_DIR,
        }
        logger.info(f"CPU ov_config: {ov_config}")

    logger.info("Loading model...")
    model = OVModelForVisualCausalLM.from_pretrained(MODEL_PATH, device=DEVICE, ov_config=ov_config)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")
    
    # Clear sampling defaults from the model config to avoid warnings when do_sample=False.
    # We will pass the requested top_p/top_k explicitly from the request schema.
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False
    
    try:
        # Report what device the model actually compiled to
        actual_device = getattr(model, '_device', None) or getattr(model.model, 'request', None)
        logger.info(f"Model compiled for device (requested): {DEVICE}")
        logger.info(f"OV_CPU_BACKEND_NUM_THREADS env: {os.getenv('OV_CPU_BACKEND_NUM_THREADS', 'not set')}")
    except Exception:
        pass  # best-effort

_initialize_model()

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
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

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
    top_p:                 Optional[float] = None
    top_k:                 Optional[int]   = None
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
FORCE_WORD_STREAM = os.getenv("FORCE_WORD_STREAM", "").lower() in ("1", "true", "yes")

# Semaphore is None until the lifespan initializes it
_inference_semaphore: asyncio.Semaphore | None = None

def _warmup_inference():
    """
    Runs a minimal generation to trigger OpenVINO JIT/shader compilation.
    Result is cached in OPENVINO_CACHE_DIR. Called once at startup.
    """
    try:
        logger.info(f"Running warmup inference on {DEVICE} to pre-compile graphs...")
        t0 = time.time()
        warmup_msgs = [{"role":"user","content":[{"type":"text","text":"Hi"}]}]
        tmpl   = processor.apply_chat_template(warmup_msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=tmpl, return_tensors="pt")
        _      = model.generate(**inputs, do_sample=False, max_new_tokens=3)
        elapsed = time.time() - t0
        logger.info(f"Warmup complete in {elapsed:.1f}s — graphs compiled and cached at {OPENVINO_CACHE_DIR}")
    except Exception as e:
        logger.warning(f"Warmup inference failed (non-fatal, will compile on first request): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_semaphore
    _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    logger.info(f"Inference semaphore initialized (max_concurrent={MAX_CONCURRENT_REQUESTS}, max_queued={MAX_QUEUED_REQUESTS})")
    # Run warmup directly on MainThread
    _warmup_inference()
    yield

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Gemma 4 OpenVINO OpenAI API", version="1.0.0", lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "available_devices": available_devices_list,
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

@app.get("/version")
def get_version():
    """Version endpoint — used by clients for server type detection."""
    return {"version": "1.0.0"}

@app.get("/api/tags")
def api_tags():
    """Ollama-compatible model listing endpoint."""
    return {
        "models": [{
            "name":        MODEL_ID,
            "model":       MODEL_ID,
            "modified_at": "2025-05-09T00:00:00Z",
            "size":        8_380_000_000,   # approximate INT8 model size in bytes
            "digest":      SYSTEM_FINGERPRINT,
            "details": {
                "parent_model":   "",
                "format":         "openvino",
                "family":         "gemma4",
                "families":       ["gemma4"],
                "parameter_size": "4B",
                "quantization_level": "INT8",
            },
        }]
    }

@app.get("/api/v1/models")
def api_v1_models():
    """Alternative model listing path probed by some Ollama-compatible clients."""
    # Delegate to the main listing
    return list_models()

@app.get("/v1/props")
@app.get("/props")
def get_props():
    """
    Model properties endpoint.
    Hermes and similar clients read this to configure streaming, context limits,
    and feature support. A 404 here causes capability fall-backs that can
    disable streaming display.
    """
    GEMMA4_MAX_CONTEXT = MAX_INPUT_TOKENS
    return {
        "default_model":       MODEL_ID,
        "total_vram":          0,                    # CPU-only inference
        "context_window":      GEMMA4_MAX_CONTEXT,
        "max_context_length":  GEMMA4_MAX_CONTEXT,
        "supports_thinking":   THINKING_SUPPORTED,
        "supports_vision":     True,
        "streaming":           True,
        "system_fingerprint":  SYSTEM_FINGERPRINT,
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

    logger.info(f"Inputs keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            logger.info(f"Input key '{k}' shape: {v.shape}, tensor: {v.tolist()}")

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
    logger.info(
        f"[_run_inference] max_new_tokens={max_new_tokens} do_sample={do_sample} "
        f"enable_thinking={enable_thinking} temperature={temperature} top_p={top_p} "
        f"top_k={top_k} repetition_penalty={repetition_penalty} messages={messages}"
    )
    inputs, input_len = _prepare_inputs(messages, enable_thinking)

    gen_params = dict(
        do_sample        = do_sample,
        max_new_tokens   = max_new_tokens,
    )
    if do_sample and temperature is not None and temperature > 0.0:
        gen_params["temperature"] = temperature
    if top_p is not None and do_sample:
        gen_params["top_p"] = top_p
    if top_k is not None and do_sample:
        gen_params["top_k"] = top_k
    if repetition_penalty is not None:
        gen_params["repetition_penalty"] = repetition_penalty

    logger.info(f"[_run_inference] model.generate inputs keys: {list(inputs.keys())}, gen_params: {gen_params}")
    output = model.generate(**inputs, **gen_params)
    output_len = output.shape[-1] - input_len
    finish_reason = "length" if output_len >= max_new_tokens else "stop"
    
    # We use skip_special_tokens=False to catch the thinking tags
    raw_response = processor.decode(output[0][input_len:], skip_special_tokens=False)
    raw_response = _strip_eos_tokens(raw_response)
    
    thinking, response = parse_thinking(raw_response)
    response = _clean_tags(response)
    return thinking, response, input_len, output_len, finish_reason


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
            if effective_do_sample and req.temperature is not None and req.temperature > 0.0:
                generate_kwargs["temperature"] = req.temperature
            if req.top_p is not None and effective_do_sample:
                generate_kwargs["top_p"] = req.top_p
            if req.top_k is not None and effective_do_sample:
                generate_kwargs["top_k"] = req.top_k
            if req.repetition_penalty is not None:
                generate_kwargs["repetition_penalty"] = req.repetition_penalty
            if req.stop:
                logger.warning(f"[{request_id}] stop sequences: post-hoc truncation only")

            generation_error: list[BaseException | None] = [None]

            _tokenizer = (
                getattr(processor, 'tokenizer', None)
                or getattr(processor, 'text_tokenizer', None)
                or processor
            )
            streamer = TextIteratorStreamer(
                _tokenizer,
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

            def _generate_word_chunks():
                """
                Fallback: run full generation then push decoded output in word-chunks.
                Used when TextIteratorStreamer is not progressive.
                """
                try:
                    # Strip the streamer key — generate_kwargs already has do_sample and max_new_tokens
                    clean_kwargs = {k: v for k, v in generate_kwargs.items() if k != "streamer"}
                    output = model.generate(**clean_kwargs)
                    decoded = processor.decode(
                        output[0][input_len:],
                        skip_special_tokens=False
                    )
                    decoded = _strip_eos_tokens(decoded)

                    # Push word-by-word for a streaming-like experience
                    import re as _re
                    parts = _re.split(r"(\s+)", decoded)
                    for part in parts:
                        if part:
                            _put(part)
                except Exception as exc:
                    generation_error[0] = exc
                finally:
                    _put(None)   # sentinel

            if FORCE_WORD_STREAM:
                gen_future = inference_executor.submit(_generate_word_chunks)
                bridge_thread = None
            else:
                gen_future    = inference_executor.submit(_generate)
                bridge_thread = threading.Thread(target=_bridge,   daemon=True)
            
            def _release_when_done():
                try:
                    gen_future.result()
                except Exception as exc:
                    logger.error(f"[{request_id}] Generation task failed: {exc}")
                loop.call_soon_threadsafe(_inference_semaphore.release)
                
            release_thread = threading.Thread(target=_release_when_done, daemon=True)

            thread_started = True
            if bridge_thread:
                bridge_thread.start()
            release_thread.start()

            # Yield an explicit SSE comment immediately — forces TCP flush
            yield ": stream-start\n\n"

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

    # Enable sampling only when temperature is meaningfully above zero and not the default.
    # temperature=0 or temperature=None means greedy — do NOT set do_sample=True.
    effective_do_sample = req.do_sample or (
        req.temperature is not None
        and req.temperature > 0.0
        and req.temperature != 1.0
    )
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
            media_type="text/event-stream",
            headers={
                "Cache-Control":    "no-cache, no-transform",
                "X-Accel-Buffering": "no",      # disables nginx/proxy buffering if present
                "Connection":       "keep-alive",
                "Transfer-Encoding": "chunked",
            }
        )
    else:
        _semaphore_acquired = False
        try:
            await _inference_semaphore.acquire()
            _semaphore_acquired = True

            t0 = time.time()
            try:
                logger.info("Running a test warmup inference inside request handler...")
                _warmup_inference()
            except Exception as w_err:
                logger.error(f"Warmup inside request handler failed: {w_err}")
                
            thinking, response_text, input_tokens, output_tokens, finish_reason = \
                _run_inference(
                    req.messages,
                    effective_max_tokens,
                    effective_do_sample,
                    effective_thinking,
                    req.temperature,
                    req.top_p,
                    req.top_k,
                    req.repetition_penalty,
                )
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
            if _semaphore_acquired:
                _inference_semaphore.release()
            with _queue_lock:
                _queue_counter -= 1

if __name__ == "__main__":
    import http.server
    import urllib.parse

    class StopGenerationException(Exception):
        pass

    class HTTPStreamer(TextStreamer):
        def __init__(self, tokenizer, request_handler, request_id, created, model_name, req_stop=None, **decode_kwargs):
            super().__init__(tokenizer, skip_prompt=True, **decode_kwargs)
            self.request_handler = request_handler
            self.request_id = request_id
            self.created = created
            self.model_name = model_name
            self.req_stop = req_stop
            self.generated_tokens = 0
            self.stop_triggered = False
            self.in_thinking = False
            self.accumulated = ""

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if self.stop_triggered:
                return

            if text:
                self.generated_tokens += 1
                self.accumulated += text

                while True:
                    found = False
                    if not self.in_thinking:
                        for tag in THINK_START_TAGS:
                            if tag in self.accumulated:
                                pos = self.accumulated.find(tag)
                                before = _clean_tags(self.accumulated[:pos])
                                if before:
                                    chunk = _make_chunk(self.request_id, self.created, self.model_name, {'content': before})
                                    self.request_handler.send_chunk(f"data: {json.dumps(chunk)}\n\n")
                                self.in_thinking = True
                                self.accumulated = self.accumulated[pos + len(tag):]
                                chunk = _make_chunk(self.request_id, self.created, self.model_name, {'content': '<think>\n'})
                                self.request_handler.send_chunk(f"data: {json.dumps(chunk)}\n\n")
                                found = True
                                break
                    else:
                        for tag in THINK_END_TAGS:
                            if tag in self.accumulated:
                                pos = self.accumulated.find(tag)
                                before = self.accumulated[:pos]
                                if before:
                                    chunk = _make_chunk(self.request_id, self.created, self.model_name, {'content': before})
                                    self.request_handler.send_chunk(f"data: {json.dumps(chunk)}\n\n")
                                self.in_thinking = False
                                self.accumulated = self.accumulated[pos + len(tag):]
                                chunk = _make_chunk(self.request_id, self.created, self.model_name, {'content': '\n</think>\n\n'})
                                self.request_handler.send_chunk(f"data: {json.dumps(chunk)}\n\n")
                                found = True
                                break
                    if not found:
                        break

                tail_len = _safe_prefix_len(self.accumulated)
                if len(self.accumulated) > MAX_ACCUMULATE:
                    tail_len = max(tail_len, _MAX_TAG_LEN)

                if len(self.accumulated) > tail_len:
                    to_yield = self.accumulated[:-tail_len] if tail_len > 0 else self.accumulated
                    self.accumulated = self.accumulated[-tail_len:] if tail_len > 0 else ""
                    to_yield = _clean_tags(to_yield)

                    if self.req_stop:
                        truncated = _apply_stop_sequences(to_yield, self.req_stop)
                        if len(truncated) < len(to_yield):
                            self.stop_triggered = True
                            to_yield = truncated
                            if to_yield:
                                chunk = _make_chunk(self.request_id, self.created, self.model_name, {'content': to_yield})
                                self.request_handler.send_chunk(f"data: {json.dumps(chunk)}\n\n")
                            raise StopGenerationException("Stop sequence triggered")

                    if to_yield:
                        chunk = _make_chunk(self.request_id, self.created, self.model_name, {'content': to_yield})
                        self.request_handler.send_chunk(f"data: {json.dumps(chunk)}\n\n")

            if stream_end:
                if self.accumulated:
                    self.accumulated = _clean_tags(self.accumulated)
                    if self.req_stop:
                        self.accumulated = _apply_stop_sequences(self.accumulated, self.req_stop)
                    if self.accumulated:
                        chunk = _make_chunk(self.request_id, self.created, self.model_name, {'content': self.accumulated})
                        self.request_handler.send_chunk(f"data: {json.dumps(chunk)}\n\n")
    
    class OpenAIHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            logger.info("%s - - %s" % (self.address_string(), format % args))

        def _send_error(self, code, message, error_type="invalid_request_error", code_str=None):
            err_data = {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": code_str
                }
            }
            body = json.dumps(err_data).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed_path = urllib.parse.urlparse(self.path)
            if parsed_path.path == "/health":
                health_data = {
                    "status": "ok",
                    "model": MODEL_ID,
                    "device": DEVICE,
                    "available_devices": available_devices_list,
                    "thinking_supported": THINKING_SUPPORTED
                }
                body = json.dumps(health_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed_path.path == "/v1/models":
                max_ctx = MAX_INPUT_TOKENS
                models_data = {
                    "object": "list",
                    "data": [{
                        "id":             MODEL_ID,
                        "object":         "model",
                        "created":        MODEL_CREATED_AT,
                        "owned_by":       "google/openvino",
                        "context_length": max_ctx,
                        "max_context_length": max_ctx,
                        "system_fingerprint": SYSTEM_FINGERPRINT,
                    }]
                }
                body = json.dumps(models_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed_path.path == "/version":
                body = json.dumps({"version": "1.7.0"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def do_POST(self):
            parsed_path = urllib.parse.urlparse(self.path)
            if parsed_path.path == "/v1/chat/completions":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                try:
                    body_dict = json.loads(body.decode("utf-8"))
                    req = ChatCompletionRequest(**body_dict)
                except Exception as e:
                    self._send_error(400, f"Invalid request body: {e}", "invalid_request_error", "bad_request")
                    return

                request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created = int(time.time())
                logger.info(f"[{request_id}] stream={req.stream} thinking={req.enable_thinking} max_tokens={req.max_tokens}")

                if req.n and req.n != 1:
                    self._send_error(400, "Only n=1 is supported.", "invalid_request_error", "unsupported_n")
                    return

                total_chars = sum(
                    len(m.content) if isinstance(m.content, str) else sum(
                        len(p.get("text", "")) for p in m.content if isinstance(p, dict)
                    )
                    for m in req.messages
                )
                if total_chars > MAX_INPUT_TOKENS * 6:
                    self._send_error(400, f"Request too large (estimated input exceeds {MAX_INPUT_TOKENS} tokens).",
                                      "invalid_request_error", "context_length_exceeded")
                    return

                effective_do_sample = req.do_sample or (
                    req.temperature is not None
                    and req.temperature > 0.0
                    and req.temperature != 1.0
                )
                effective_max_tokens = req.max_completion_tokens or req.max_tokens or 512

                effective_thinking = (
                    (req.enable_thinking or False)
                    or (req.reasoning_effort is not None and req.reasoning_effort.lower() == "high")
                )

                if req.stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()

                    # OpenAI spec: first chunk always carries role, with empty content
                    first_chunk = _make_chunk(request_id, created, req.model, {"role": "assistant", "content": ""})
                    self.send_chunk(f"data: {json.dumps(first_chunk)}\n\n")

                    inputs, input_len = _prepare_inputs(req.messages, effective_thinking)
                    generate_kwargs = dict(
                        **inputs,
                        do_sample=effective_do_sample,
                        max_new_tokens=effective_max_tokens,
                    )
                    if effective_do_sample and req.temperature is not None and req.temperature > 0.0:
                        generate_kwargs["temperature"] = req.temperature
                    if req.top_p is not None and effective_do_sample:
                        generate_kwargs["top_p"] = req.top_p
                    if req.top_k is not None and effective_do_sample:
                        generate_kwargs["top_k"] = req.top_k
                    if req.repetition_penalty is not None:
                        generate_kwargs["repetition_penalty"] = req.repetition_penalty

                    _tokenizer = (
                        getattr(processor, 'tokenizer', None)
                        or getattr(processor, 'text_tokenizer', None)
                        or processor
                    )

                    streamer = HTTPStreamer(
                        _tokenizer,
                        self,
                        request_id,
                        created,
                        req.model,
                        req_stop=req.stop,
                        skip_special_tokens=False
                    )
                    generate_kwargs["streamer"] = streamer

                    t_start = time.time()
                    try:
                        model.generate(**generate_kwargs)
                    except StopGenerationException:
                        logger.info(f"[{request_id}] Generation stopped early via stop sequence.")
                    except Exception as e:
                        logger.error(f"[{request_id}] Streaming generation error: {e}")

                    streamer.end()

                    finish_reason = "length" if streamer.generated_tokens >= effective_max_tokens else "stop"
                    final_chunk = _make_chunk(request_id, created, req.model, {}, finish_reason)
                    self.send_chunk(f"data: {json.dumps(final_chunk)}\n\n")

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
                                "completion_tokens": streamer.generated_tokens,
                                "total_tokens":      input_len + streamer.generated_tokens,
                            },
                        }
                        self.send_chunk(f"data: {json.dumps(usage_chunk)}\n\n")

                    self.send_chunk("data: [DONE]\n\n")
                    self.end_chunks()

                    elapsed = time.time() - t_start
                    logger.info(
                        f"[{request_id}] streamed {streamer.generated_tokens} tokens in {elapsed:.2f}s "
                        f"({streamer.generated_tokens / elapsed:.1f} tok/s)"
                    )
                else:
                    t0 = time.time()
                    try:
                        thinking, response_text, input_tokens, output_tokens, finish_reason = \
                            _run_inference(
                                req.messages,
                                effective_max_tokens,
                                effective_do_sample,
                                effective_thinking,
                                req.temperature,
                                req.top_p,
                                req.top_k,
                                req.repetition_penalty,
                            )
                    except Exception as e:
                        logger.error(f"[{request_id}] Inference failed: {e}")
                        self._send_error(500, f"Inference failed: {e}", "server_error", "internal_error")
                        return

                    elapsed = time.time() - t0
                    logger.info(f"[{request_id}] Generated {output_tokens} tokens in {elapsed:.2f}s ({output_tokens/elapsed:.1f} tok/s)")

                    response_text = _apply_stop_sequences(response_text, req.stop)
                    if thinking:
                        response_text = f"<think>\n{thinking}\n</think>\n\n{response_text}"

                    final_content = response_text.strip()
                    message = {"role": "assistant", "content": final_content}

                    resp_data = {
                        "id": request_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": req.model,
                        "system_fingerprint": SYSTEM_FINGERPRINT,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        }
                    }

                    body = json.dumps(resp_data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def send_chunk(self, data: str):
            chunk_data = data.encode("utf-8")
            chunk_len = f"{len(chunk_data):x}\r\n".encode("utf-8")
            self.wfile.write(chunk_len)
            self.wfile.write(chunk_data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def end_chunks(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    server_address = ("", 8000)
    httpd = http.server.HTTPServer(server_address, OpenAIHTTPRequestHandler)
    logger.info("Starting production synchronous HTTP server on port 8000...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
