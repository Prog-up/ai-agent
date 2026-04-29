# =============================================================================
# DeepSeek-V2-Lite — OpenVINO INT4 Conversion + Benchmark
# Intel i7 CPU + Intel Xe iGPU (32 GB RAM)
# =============================================================================
#
# DEPENDENCIES — install with uv:
#
#   uv init deepseek-bench
#   cd deepseek-bench
#
#   uv add optimum-intel[openvino] \
#           openvino-genai \
#           transformers \
#           huggingface-hub \
#           nncf \
#           rich \
#           tabulate
#
#   uv run python deepseek_ov_benchmark.py
#
# NOTES:
#   • DeepSeek-V2-Lite (~16 B params, MoE) needs ≈ 20 GB RAM for conversion.
#     Close other heavy apps before running.
#   • Conversion is skipped automatically if the output dir already exists.
#   • trust_remote_code=True is required for DeepSeek's custom MLA attention.
#   • For OpenVINO GPU, ensure Intel GPU drivers + OpenCL runtime are installed:
#     https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html
# =============================================================================

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tabulate import tabulate

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

SOURCE_MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
OV_OUTPUT_DIR   = Path("./models/DeepSeek-V2-Lite-int4-ov")

BENCHMARK_PROMPT = (
    "You are a helpful assistant. Explain the key differences between "
    "transformer-based language models and recurrent neural networks, "
    "covering architecture, training, and inference characteristics in detail."
)

MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.0   # greedy / deterministic

# =============================================================================
# STEP 1 — CONVERT TO OPENVINO INT4
# =============================================================================

def convert_model():
    """
    Download DeepSeek-V2-Lite from HuggingFace and export it to OpenVINO IR
    with INT4 weight-only quantization (AWQ-style, asymmetric, ratio=0.8).

    ratio=0.8 means 80 % of weights are INT4, the remaining 20 % stay in FP16
    (applied to the most sensitive layers) — a good accuracy/size trade-off for
    large MoE models.

    Skips conversion entirely if the output directory already exists and is
    non-empty (idempotent re-runs).
    """
    if OV_OUTPUT_DIR.exists() and any(OV_OUTPUT_DIR.iterdir()):
        console.print(f"[green]✓ Converted model already exists:[/green] {OV_OUTPUT_DIR}")
        return

    console.rule("[bold yellow]Converting DeepSeek-V2-Lite → OpenVINO INT4")
    console.print(f"  Source : [cyan]{SOURCE_MODEL_ID}[/cyan]")
    console.print(f"  Output : [cyan]{OV_OUTPUT_DIR}[/cyan]")
    console.print("  [dim]This will download ~30 GB and may take 20-60 min on first run.[/dim]\n")

    # Lazy imports — only needed during conversion, not benchmarking
    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
    from transformers import AutoTokenizer

    OV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # INT4 asymmetric weight quantization.
    # group_size=128 is the standard granularity for 4-bit LLM quantization.
    quant_config = OVWeightQuantizationConfig(
        bits=4,
        asym=True,       # asymmetric (unsigned INT4) — better accuracy than sym
        ratio=0.8,       # 80 % INT4 / 20 % FP16 for sensitive layers
        group_size=128,
    )

    console.print("[dim]Loading and exporting model (this allocates ~20 GB RAM)…[/dim]")
    t0 = time.perf_counter()

    model = OVModelForCausalLM.from_pretrained(
        SOURCE_MODEL_ID,
        export=True,                        # triggers ONNX → OV IR conversion
        quantization_config=quant_config,
        trust_remote_code=True,             # required for DeepSeek MLA attention
        # compile=False defers OpenVINO compilation to load time so we can save
        # the IR files first without triggering a device-specific compile.
        compile=False,
    )

    console.print("[dim]Saving OpenVINO IR weights and config…[/dim]")
    model.save_pretrained(str(OV_OUTPUT_DIR))

    console.print("[dim]Saving tokenizer…[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_MODEL_ID, trust_remote_code=True
    )
    tokenizer.save_pretrained(str(OV_OUTPUT_DIR))

    elapsed = time.perf_counter() - t0
    console.print(
        f"\n[bold green]✓ Conversion complete[/bold green] "
        f"in {elapsed/60:.1f} min → {OV_OUTPUT_DIR}"
    )

    # Free memory before benchmarking starts
    del model, tokenizer

# =============================================================================
# STEP 2 — OPENVINO GENAI BENCHMARK
# =============================================================================

def run_openvino_benchmark(device: str, warmup: bool = True) -> dict:
    """
    Load the converted OV model with openvino_genai.LLMPipeline and run one
    timed inference pass, using a streaming callback to capture per-token
    timestamps for accurate TTFT and decode-speed measurement.

    OpenVINO GenAI streamer protocol (important!):
        return False  →  continue generation   ✅
        return True   →  STOP generation       🛑
    """
    import openvino_genai as ov_genai

    console.print(f"  [dim]Loading OV pipeline on {device}…[/dim]")
    pipe = ov_genai.LLMPipeline(str(OV_OUTPUT_DIR), device)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample      = False   # greedy decoding
    try:
        config.temperature = TEMPERATURE
    except AttributeError:
        pass

    def _run_once() -> dict:
        token_times = []
        wall_start  = time.perf_counter()

        # Called once per generated subword/token string.
        # MUST return False to continue; True means stop.
        def streamer_callback(subword: str) -> bool:
            token_times.append(time.perf_counter())
            return False  # False = continue generation

        pipe.generate(BENCHMARK_PROMPT, config, streamer=streamer_callback)

        wall_end = time.perf_counter()

        if not token_times:
            raise RuntimeError("No tokens were generated — check the model.")

        ttft          = token_times[0] - wall_start
        total_latency = wall_end - wall_start
        decode_tokens = len(token_times) - 1          # exclude first (prefill) token
        decode_time   = wall_end - token_times[0]
        decode_speed  = decode_tokens / decode_time if decode_time > 0 else 0.0

        return {
            "backend":         "openvino_genai",
            "model":           "DeepSeek-V2-Lite-INT4",
            "device":          device,
            "ttft_s":          round(ttft, 4),
            "decode_tps":      round(decode_speed, 2),
            "total_tokens":    len(token_times),
            "total_latency_s": round(total_latency, 4),
        }

    if warmup:
        console.print(f"  [dim]Warming up on {device}…[/dim]")
        _run_once()   # discard — populates KV-cache, JIT, OpenCL kernels

    console.print(f"  [bold]Running benchmark:[/bold] DeepSeek-V2-Lite / {device}")
    result = _run_once()

    del pipe   # release VRAM / system RAM before next device run

    return result

# =============================================================================
# STEP 3 — DISPLAY RESULTS
# =============================================================================

def display_results(results: list[dict]):
    if not results:
        console.print("[red]No results to display.[/red]")
        return

    console.rule("[bold green]Benchmark Results")

    # Rich table
    table = Table(
        show_header=True,
        header_style="bold magenta",
        title="DeepSeek-V2-Lite INT4 — OpenVINO Benchmark (Intel i7 + Xe iGPU)",
    )
    table.add_column("Backend",        style="cyan",  no_wrap=True)
    table.add_column("Model",          style="white", no_wrap=False, max_width=30)
    table.add_column("Device",         style="yellow", justify="center")
    table.add_column("TTFT (s)",       style="green",  justify="right")
    table.add_column("Decode (tok/s)", style="green",  justify="right")
    table.add_column("Total Tokens",   style="blue",   justify="right")
    table.add_column("Latency (s)",    style="red",    justify="right")

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

    # Plain-text fallback (easy to paste elsewhere)
    headers = ["Backend", "Model", "Device", "TTFT(s)",
               "Decode(tok/s)", "TotalTok", "Latency(s)"]
    rows = [
        [r["backend"], r["model"], r["device"],
         r["ttft_s"], r["decode_tps"], r["total_tokens"], r["total_latency_s"]]
        for r in results
    ]
    console.print("\n[dim]Plain-text table (copy-friendly):[/dim]")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # JSON export
    output_file = Path("deepseek_benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "meta": {
                    "source_model":   SOURCE_MODEL_ID,
                    "ov_model_dir":   str(OV_OUTPUT_DIR),
                    "quantization":   "INT4-asym ratio=0.8 group_size=128",
                    "prompt_chars":   len(BENCHMARK_PROMPT),
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature":    TEMPERATURE,
                    "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
    console.rule("[bold blue]DeepSeek-V2-Lite — OpenVINO Conversion + Benchmark")
    console.print(f"Source model  : [cyan]{SOURCE_MODEL_ID}[/cyan]")
    console.print(f"OV output dir : [cyan]{OV_OUTPUT_DIR}[/cyan]")
    console.print(f"Prompt length : {len(BENCHMARK_PROMPT)} chars")
    console.print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    console.print(f"temperature   : {TEMPERATURE}  (greedy decoding)\n")

    # 1. Convert (skipped if already done)
    convert_model()

    # 2. Benchmark on CPU then GPU
    results = []
    console.rule("[bold yellow]OpenVINO GenAI — DeepSeek-V2-Lite")

    for device in ("CPU", "GPU"):
        try:
            r = run_openvino_benchmark(device=device, warmup=True)
            results.append(r)
        except Exception as e:
            console.print(f"[red]✗ {device} run failed: {e}[/red]")

    # 3. Print table + write JSON
    display_results(results)


if __name__ == "__main__":
    main()
