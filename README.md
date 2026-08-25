# MiniCPM-o 4.5 × LongSpeech

Streaming evaluation harness for **MiniCPM-o 4.5** on the LongSpeech test sets:

- `ASR`
- `summary`
- `Temporal_Relative_QA`

The default inference path is intentionally **half-duplex streaming**: the benchmark instruction is prefetched, 16 kHz audio is fed as 1-second chunks through `streaming_prefill()`, and text is produced once at the end through `streaming_generate()`.

## RunPod target

Target image:

```text
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404
```

The base image ships a newer PyTorch than MiniCPM-o 4.5's documented Transformers streaming stack. This project therefore uses an isolated `uv` environment and pins:

- Python 3.10
- `torch==2.8.0`
- `torchaudio==2.8.0`
- `transformers==4.51.0`
- `minicpmo-utils[all]>=1.0.5`

## Setup

```bash
git clone https://github.com/lyh030725/minicpm-longspeech.git
cd minicpm-longspeech
bash scripts/setup.sh
```

Or manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync
```

For Hugging Face authentication if required:

```bash
export HF_TOKEN=...
```

To keep the large model/dataset cache on a RunPod Network Volume:

```bash
export HF_HOME=/workspace/hf_cache
```

## Smoke test first

```bash
uv run python run.py --task ASR --limit 1
```

Then:

```bash
uv run python run.py --task ASR --limit 10
```

Predictions are appended sample-by-sample, so a crash/OOM does not discard previous results.

## Full evaluation

```bash
uv run python run.py --task ASR --resume
uv run python run.py --task summary --resume
uv run python run.py --task Temporal_Relative_QA --resume
```

or:

```bash
bash scripts/run_all.sh
```

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

The repository does **not** clone the roughly 2 TB LongSpeech dataset. At runtime it downloads only the task JSONL and resolves referenced WAVs lazily through Hugging Face Hub, reusing the Hub cache.

Pre-download a task:

```bash
uv run python scripts/download_data.py --task summary
```

Metadata only:

```bash
uv run python scripts/download_data.py --task ASR --no-audio
```

Small subset:

```bash
uv run python scripts/download_data.py --task ASR --limit 10
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
uv run python run.py --task ASR --start-index 0 --end-index 1000
```

Use separate `--output-root` directories for concurrently running shards, then concatenate JSONL files after completion.

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

The runner writes language-wise error rates: CJK languages use CER; other languages use WER. It also records an overall reference-unit-weighted error rate.

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
uv run python run.py --task summary --no-resume
uv run python run.py --task ASR --attn-implementation flash_attention_2
```

`flash_attention_2` requires a compatible FlashAttention installation. The default `sdpa` avoids that extra dependency.

## OOM / failure behavior

Every sample is wrapped independently. Failures are written to the prediction JSONL. Successful earlier samples remain intact. `--resume` skips sample IDs already present in the output.

For a 24 GB GPU, start with one real LongSpeech sample and inspect `peak_vram_allocated_mb`, `peak_vram_reserved_mb`, `prefill_seconds`, and `generation_seconds`.

## Reproducibility notes

- No custom system prompt is added.
- The LongSpeech user instruction is used verbatim.
- Sampling is disabled.
- Each sample gets a fresh MiniCPM streaming session and model session reset.
- Task JSONL and WAVs default to dataset `main`; set `--dataset-revision <commit>` to freeze a paper run.
