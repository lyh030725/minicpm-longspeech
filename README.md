# MiniCPM-o 4.5 × LongSpeech

Streaming evaluation harness for **MiniCPM-o 4.5** on LongSpeech:

- `ASR`: first **1,000** samples from the test split
- `summary`: **all** test samples
- `Temporal_Relative_QA`: **all** test samples

The default inference path is intentionally **half-duplex streaming**: the benchmark instruction is prefetched, 16 kHz audio is fed as 1-second chunks through `streaming_prefill()`, and text is produced once at the end through `streaming_generate()`.

## RunPod target

Use this RunPod image:

```text
runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster
```

This matches the MiniCPM-o 4.5 streaming stack much more closely than a PyTorch 2.9 image. The project uses an isolated `uv` environment with:

- Python 3.10
- `torch==2.8.0` from the official PyTorch **cu128** wheel index
- `torchaudio==2.8.0` from the official PyTorch **cu128** wheel index
- `transformers==4.51.0`
- `accelerate==1.12.0`
- `minicpmo-utils[all]>=1.0.5`

`pyproject.toml` explicitly pins `torch` and `torchaudio` to `https://download.pytorch.org/whl/cu128`, while normal Python dependencies continue to resolve from PyPI. This prevents `uv` from accidentally selecting a CPU wheel or a different CUDA build.

The RunPod image already contains a compatible system PyTorch, but the project deliberately keeps its own `.venv` so the evaluation environment is reproducible and managed entirely by `uv`.

## Setup

```bash
git clone https://github.com/lyh030725/minicpm-longspeech.git
cd minicpm-longspeech
bash scripts/setup.sh
```

The setup script:

1. installs `uv` if necessary,
2. installs Python 3.10,
3. runs `uv sync`,
4. verifies Python, PyTorch, CUDA build, torchaudio, Transformers, and GPU access.

A successful setup should report values equivalent to:

```text
Python       : 3.10.x
PyTorch      : 2.8.0+cu128
Torch CUDA   : 12.8
Torchaudio   : 2.8.0+cu128
Transformers : 4.51.0
CUDA usable  : True
```

Or install manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync
```

For Hugging Face authentication if required:

```bash
export HF_TOKEN=...
```

To keep the model/dataset cache on a RunPod Network Volume:

```bash
export HF_HOME=/workspace/hf_cache
```

## Evaluation scope

The project defaults are intentionally task-specific:

| Task | Default test scope |
|---|---:|
| ASR | first 1,000 samples |
| summary | entire split |
| Temporal_Relative_QA | entire split |

Therefore these commands already use the requested evaluation sizes:

```bash
uv run python run.py --task ASR --resume
uv run python run.py --task summary --resume
uv run python run.py --task Temporal_Relative_QA --resume
```

`--limit N` overrides the default for a particular run. `--all-samples` disables the task-specific limit, including the ASR 1,000-sample cap.

## Smoke test first

```bash
uv run python run.py --task ASR --limit 1
```

Then:

```bash
uv run python run.py --task ASR --limit 10
```

Predictions are appended sample-by-sample, so a crash/OOM does not discard previous results.

## Full requested evaluation

```bash
bash scripts/run_all.sh
```

This runs ASR top 1,000 plus the full summary and Temporal Relative QA test sets.

Outputs:

```text
outputs/
├── ASR/
│   ├── test.predictions.jsonl
│   └── test.predictions.metrics.json
├── summary/
│   ├── test.predictions.jsonl
│   └── test.predictions.metrics.json
└── Temporal_Relative_QA/
    └── test.predictions.jsonl
```

Each prediction row includes the reference/prediction plus audio duration, number of chunks, prefill/generation latency, prompt position, and CUDA peak memory.

## Dataset download behavior

The repository does **not** clone the entire LongSpeech dataset. It downloads the task JSONLs and only the referenced WAVs needed for the configured evaluation scope, while reusing the Hugging Face Hub cache.

To pre-download exactly the requested evaluation set in one command:

```bash
uv run python scripts/download_data.py
```

That means:

```text
ASR                  first 1,000 test WAVs
summary              all test WAVs
Temporal_Relative_QA all test WAVs
```

Download one task only:

```bash
uv run python scripts/download_data.py --task summary
```

Metadata only:

```bash
uv run python scripts/download_data.py --no-audio
```

Small subset override:

```bash
uv run python scripts/download_data.py --task ASR --limit 10
```

Full ASR override:

```bash
uv run python scripts/download_data.py --task ASR --all-samples
```

## Streaming definition

Default:

```text
benchmark prompt
    ↓
streaming_prefill(text)
    ↓
audio chunk 0 (1 s)
    ↓
audio chunk 1 (1 s)
    ↓
...
    ↓
last audio chunk (is_last_chunk=True)
    ↓
streaming_generate(text only)
```

Defaults:

```text
sample rate       16,000 Hz
chunk size        1.0 s
omni_mode         False
generate_audio    False
use_tts_template  False
enable_thinking   False
do_sample         False
vision            disabled
TTS               disabled
```

The final short audio chunk is zero-padded to at least 16,000 samples, matching the official MiniCPM-o 4.5 streaming example.

### Prompt order ablation

Default:

```bash
--prompt-position before
```

Audio-then-instruction variant:

```bash
uv run python run.py --task Temporal_Relative_QA --prompt-position after
```

The selected value is stored in every result row.

## Sharding

```bash
uv run python run.py --task ASR --start-index 0 --end-index 500 --output-root outputs/shard0
uv run python run.py --task ASR --start-index 500 --end-index 1000 --output-root outputs/shard1
```

Use separate output directories for concurrently running shards, then concatenate JSONL files after completion.

## Generation lengths

| Task | `max_new_tokens` |
|---|---:|
| ASR | 8192 |
| summary | 1024 |
| Temporal_Relative_QA | 256 |

Override:

```bash
uv run python run.py --task ASR --max-new-tokens 12000
```

## Metrics

### ASR

The runner reports:

- Non-CJK overall WER
- CJK overall CER
- Overall CER across all languages
- per-language WER/CER

### Summarization

The runner writes ROUGE-1 F1, ROUGE-2 F1, and ROUGE-L F1.

### Temporal Relative QA

LongSpeech evaluates temporal localization with an LLM judge that assigns `YES`, `PARTIALLY`, or `NO`. Strict accuracy is `YES / N`; relaxed accuracy is `(YES + PARTIALLY) / N`.

Inference is separated from judging. First finish MiniCPM-o predictions, then:

```bash
uv sync --extra judge
export OPENAI_API_KEY=...
uv run longspeech-eval judge-temporal \
  --predictions outputs/Temporal_Relative_QA/test.predictions.jsonl \
  --model gpt-4-turbo
```

For another OpenAI-compatible endpoint:

```bash
uv run longspeech-eval judge-temporal \
  --predictions outputs/Temporal_Relative_QA/test.predictions.jsonl \
  --model YOUR_MODEL \
  --base-url https://your-endpoint/v1
```

Judgments and metrics are saved separately so inference never has to be rerun.

## Useful options

```bash
uv run python run.py --help
uv run python run.py --task ASR --limit 5
uv run python run.py --task ASR --all-samples
uv run python run.py --task summary --no-resume
uv run python run.py --task ASR --attn-implementation flash_attention_2
```

`flash_attention_2` requires a compatible FlashAttention installation. The default `sdpa` avoids that extra dependency.

## OOM / failure behavior

Every sample is wrapped independently. Failures are written to the prediction JSONL. Successful earlier samples remain intact. `--resume` skips sample IDs already present in the output.

LongSpeech inputs are much longer than normal inference prompts, so a GPU with more VRAM than the model's bare loading requirement is strongly preferred. Start with one real sample and inspect `peak_vram_allocated_mb`, `peak_vram_reserved_mb`, `prefill_seconds`, and `generation_seconds` before launching the full evaluation.

## Reproducibility notes

- No custom system prompt is added.
- The LongSpeech user instruction is used verbatim.
- Sampling is disabled.
- Each sample gets a fresh MiniCPM streaming session and model session reset.
- Task JSONL and WAVs default to dataset `main`; set `--dataset-revision <commit>` to freeze a paper run.
