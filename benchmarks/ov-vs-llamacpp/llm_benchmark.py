# =============================================================================
# LLM INFERENCE BENCHMARK: OpenVINO GenAI vs llama-cpp-python
# Intel i7 CPU + Intel Xe iGPU (32 GB RAM)
# =============================================================================
#
# DEPENDENCIES — install with uv:
#
#   uv init llm-bench
#   cd llm-bench
#
#   uv add openvino-genai
#   uv add llama-cpp-python \
#       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
#       # ↑ swap the index URL for a Vulkan or SYCL wheel if you have one,
#       #   e.g. https://abetlen.github.io/llama-cpp-python/whl/vulkan
#
#   uv add huggingface-hub rich tabulate
#
#   uv run python llm_benchmark.py
#
# NOTE: llama-cpp-python GPU (Vulkan) wheel example:
#   CMAKE_ARGS="-DGGML_VULKAN=on" uv add llama-cpp-python --no-binary llama-cpp-python
#
# NOTE: For OpenVINO GPU support, ensure the Intel GPU drivers + OpenCL runtime
#   are installed on your system:
#   https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html
# =============================================================================

import json
import time
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from tabulate import tabulate

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

BENCHMARK_PROMPT = (
    "You are a helpful assistant. Explain the key differences between "
    "transformer-based language models and recurrent neural networks, "
    "covering architecture, training, and inference characteristics in detail."
)

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0  # greedy / deterministic

# Where to cache downloaded models locally
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

# ---------- Model definitions -----------------------------------------------

OV_MISTRAL_REPO = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov"
OV_MIXTRAL_REPO = "OpenVINO/mixtral-8x7b-instruct-v0.1-int4-ov"

GGUF_MISTRAL_REPO = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
GGUF_MISTRAL_FILE = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

GGUF_MIXTRAL_REPO = "TheBloke/Mixtral-8x7B-v0.1-GGUF"
GGUF_MIXTRAL_FILE = "mixtral-8x7b-v0.1.Q4_K_M.gguf"

# =============================================================================
# STEP 1 — MODEL DOWNLOAD
# =============================================================================


def download_ov_model(repo_id: str) -> Path:
    """Download a full OpenVINO model snapshot (weights + tokenizer + config)."""
    local_dir = MODELS_DIR / repo_id.replace("/", "--")
    if local_dir.exists() and any(local_dir.iterdir()):
        console.print(f"[green]✓ OV model already cached:[/green] {local_dir}")
        return local_dir
    console.print(f"[cyan]⬇ Downloading OV model:[/cyan] {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    console.print(f"[green]✓ Saved to:[/green] {local_dir}")
    return local_dir


def download_gguf_model(repo_id: str, filename: str) -> Path:
    """Download a single GGUF file."""
    dest = MODELS_DIR / filename
    if dest.exists():
        console.print(f"[green]✓ GGUF already cached:[/green] {dest}")
        return dest
    console.print(f"[cyan]⬇ Downloading GGUF:[/cyan] {repo_id}/{filename}")
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=str(MODELS_DIR)
    )
    console.print(f"[green]✓ Saved to:[/green] {path}")
    return Path(path)


def download_all_models():
    console.rule("[bold yellow]Downloading Models")
    paths = {}
    paths["ov_mistral"] = download_ov_model(OV_MISTRAL_REPO)
    paths["ov_mixtral"] = download_ov_model(OV_MIXTRAL_REPO)
    paths["gguf_mistral"] = download_gguf_model(GGUF_MISTRAL_REPO, GGUF_MISTRAL_FILE)
    # paths["gguf_mixtral"] = download_gguf_model(GGUF_MIXTRAL_REPO, GGUF_MIXTRAL_FILE)
    return paths


# =============================================================================
# STEP 2 — OPENVINO GENAI BENCHMARK
# =============================================================================


def run_openvino_benchmark(
    model_dir: Path, device: str, model_name: str, warmup: bool = True
) -> dict:
    """
    Run a single OpenVINO GenAI inference and return timing metrics.

    Uses the streaming generator to accurately split TTFT from decode speed:
      - TTFT  = wall-time from generate() call until first token arrives
      - Decode speed = (total_tokens - 1) / (total_time - ttft)
    """
    import openvino_genai as ov_genai  # imported lazily so the script doesn't
    # crash if the package isn't installed

    console.print(f"  [dim]Loading OV pipeline: {model_name} on {device}[/dim]")
    pipe = ov_genai.LLMPipeline(str(model_dir), device)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample = False  # greedy decoding → temperature=0
    # OpenVINO GenAI uses do_sample=False for greedy; temperature is ignored
    # but set it explicitly if the API version supports it:
    try:
        config.temperature = TEMPERATURE
    except AttributeError:
        pass

    def _run_once() -> dict:
        token_times = []
        wall_start = time.perf_counter()

        # Streaming callback — called once per generated token (decoded text chunk).
        # IMPORTANT: in OpenVINO GenAI, returning True means STOP generation;
        # returning False means CONTINUE.  Always return False here.
        def token_callback(subword: str) -> bool:
            token_times.append(time.perf_counter())
            return False  # False = continue;  True = stop (do NOT return True)

        # generate_stream blocks until generation is complete;
        # our callback fires for every decoded token.
        pipe.generate(BENCHMARK_PROMPT, config, streamer=token_callback)

        wall_end = time.perf_counter()

        if not token_times:
            raise RuntimeError("No tokens were generated — check the model.")

        ttft = token_times[0] - wall_start
        total_latency = wall_end - wall_start
        decode_tokens = len(token_times) - 1  # exclude first token
        decode_time = wall_end - token_times[0]  # time after first token
        decode_speed = decode_tokens / decode_time if decode_time > 0 else 0.0

        return {
            "backend": "openvino_genai",
            "model": model_name,
            "device": device,
            "ttft_s": round(ttft, 4),
            "decode_tps": round(decode_speed, 2),
            "total_tokens": len(token_times),
            "total_latency_s": round(total_latency, 4),
        }

    if warmup:
        console.print(f"  [dim]Warming up {model_name} / {device} …[/dim]")
        _run_once()  # discard warmup result

    console.print(f"  [bold]Running benchmark:[/bold] {model_name} / {device}")
    result = _run_once()

    # Free the pipeline to reclaim memory before the next test
    del pipe

    return result


# =============================================================================
# STEP 3 — LLAMA-CPP-PYTHON BENCHMARK
# =============================================================================


def run_llamacpp_benchmark(
    gguf_path: Path, device: str, model_name: str, warmup: bool = True
) -> dict:
    """
    Run a single llama-cpp-python inference and return timing metrics.

    device="CPU"  → n_gpu_layers=0
    device="GPU"  → n_gpu_layers=-1 (offload all layers; requires Vulkan/SYCL build)

    Streaming is enabled via stream=True on __call__, which yields individual
    token dicts so we can measure TTFT precisely.
    """
    from llama_cpp import Llama  # lazy import

    n_gpu_layers = -1 if device == "GPU" else 0

    console.print(f"  [dim]Loading llama.cpp model: {model_name} on {device}[/dim]")
    llm = Llama(
        model_path=str(gguf_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=2048,
        verbose=False,
    )

    def _run_once() -> dict:
        token_times = []
        wall_start = time.perf_counter()

        # stream=True returns a generator; iterate it token by token
        stream = llm(
            BENCHMARK_PROMPT,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
            echo=False,
        )

        for chunk in stream:
            token_times.append(time.perf_counter())
            # chunk["choices"][0]["text"] holds the token text if needed

        wall_end = time.perf_counter()

        if not token_times:
            raise RuntimeError("No tokens generated — check the model / prompt.")

        ttft = token_times[0] - wall_start
        total_latency = wall_end - wall_start
        decode_tokens = len(token_times) - 1
        decode_time = wall_end - token_times[0]
        decode_speed = decode_tokens / decode_time if decode_time > 0 else 0.0

        return {
            "backend": "llama_cpp",
            "model": model_name,
            "device": device,
            "ttft_s": round(ttft, 4),
            "decode_tps": round(decode_speed, 2),
            "total_tokens": len(token_times),
            "total_latency_s": round(total_latency, 4),
        }

    if warmup:
        console.print(f"  [dim]Warming up {model_name} / {device} …[/dim]")
        _run_once()

    console.print(f"  [bold]Running benchmark:[/bold] {model_name} / {device}")
    result = _run_once()

    del llm

    return result


# =============================================================================
# STEP 4 — ORCHESTRATE ALL TESTS
# =============================================================================


def run_all_benchmarks(model_paths: dict) -> list[dict]:
    results = []

    # console.rule("[bold yellow]OpenVINO GenAI — Mistral-7B")
    # for device in ("CPU", "GPU"):
    #     try:
    #         r = run_openvino_benchmark(
    #             model_dir=model_paths["ov_mistral"],
    #             device=device,
    #             model_name="Mistral-7B-Instruct-v0.3-INT4",
    #         )
    #         results.append(r)
    #     except Exception as e:
    #         console.print(f"[red]✗ OV Mistral {device} failed: {e}[/red]")

    console.rule("[bold yellow]OpenVINO GenAI — Mixtral-8x7B")
    for device in ("CPU", "GPU"):
        try:
            r = run_openvino_benchmark(
                model_dir=model_paths["ov_mixtral"],
                device=device,
                model_name="Mixtral-8x7B-Instruct-v0.1-INT4",
            )
            results.append(r)
        except Exception as e:
            console.print(f"[red]✗ OV Mixtral {device} failed: {e}[/red]")

    # console.rule("[bold yellow]llama.cpp — Mistral-7B")
    # for device in ("CPU", "GPU"):
    #     try:
    #         r = run_llamacpp_benchmark(
    #             gguf_path=model_paths["gguf_mistral"],
    #             device=device,
    #             model_name="Mistral-7B-Instruct-v0.3-Q4_K_M",
    #         )
    #         results.append(r)
    #     except Exception as e:
    #         console.print(f"[red]✗ llama.cpp Mistral {device} failed: {e}[/red]")

    # console.rule("[bold yellow]llama.cpp — Mixtral-8x7B")
    # for device in ("CPU", "GPU"):
    #     try:
    #         r = run_llamacpp_benchmark(
    #             gguf_path  = model_paths["gguf_mixtral"],
    #             device     = device,
    #             model_name = "Mixtral-8x7B-v0.1-Q4_K_M",
    #         )
    #         results.append(r)
    #     except Exception as e:
    #         console.print(f"[red]✗ llama.cpp Mixtral {device} failed: {e}[/red]")

    return results


# =============================================================================
# STEP 5 — OUTPUT: RICH TABLE + JSON FILE
# =============================================================================


def display_results(results: list[dict]):
    if not results:
        console.print("[red]No results to display.[/red]")
        return

    console.rule("[bold green]Benchmark Results")

    # --- Rich table -----------------------------------------------------------
    table = Table(
        show_header=True,
        header_style="bold magenta",
        title="LLM Inference Benchmark — Intel i7 + Xe iGPU",
    )

    table.add_column("Backend", style="cyan", no_wrap=True)
    table.add_column("Model", style="white", no_wrap=False, max_width=36)
    table.add_column("Device", style="yellow", justify="center")
    table.add_column("TTFT (s)", style="green", justify="right")
    table.add_column("Decode (tok/s)", style="green", justify="right")
    table.add_column("Total Tokens", style="blue", justify="right")
    table.add_column("Latency (s)", style="red", justify="right")

    for r in results:
        table.add_row(
            r["backend"],
            r["model"],
            r["device"],
            str(r["ttft_s"]),
            str(r["decode_tps"]),
            str(r["total_tokens"]),
            str(r["total_latency_s"]),
        )

    console.print(table)

    # --- tabulate fallback (plain text, easy to copy) -------------------------
    headers = [
        "Backend",
        "Model",
        "Device",
        "TTFT(s)",
        "Decode(tok/s)",
        "TotalTok",
        "Latency(s)",
    ]
    rows = [
        [
            r["backend"],
            r["model"],
            r["device"],
            r["ttft_s"],
            r["decode_tps"],
            r["total_tokens"],
            r["total_latency_s"],
        ]
        for r in results
    ]
    console.print("\n[dim]Plain-text table (copy-friendly):[/dim]")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # --- JSON export ----------------------------------------------------------
    output_file = Path("benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "meta": {
                    "prompt_chars": len(BENCHMARK_PROMPT),
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": TEMPERATURE,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    console.print(
        f"\n[bold green]✓ Results saved to:[/bold green] {output_file.resolve()}"
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    console.rule("[bold blue]LLM Inference Benchmark")
    console.print(f"Prompt length : {len(BENCHMARK_PROMPT)} chars")
    console.print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    console.print(f"temperature   : {TEMPERATURE}  (greedy decoding)")
    console.print()

    # 1. Download models (idempotent — skips if already cached)
    model_paths = download_all_models()

    # 2. Run all backend × device × model combinations
    results = run_all_benchmarks(model_paths)

    # 3. Print table and write JSON
    display_results(results)


if __name__ == "__main__":
    main()
