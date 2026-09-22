# Reproducing State-of-Thought

## Reproducibility contract

Every released result is tied to the following immutable chain:

```text
backbone + precision -> ordered dataset IDs -> controller -> inference config
-> grader -> per-example records -> aggregate
```

Four machine-readable manifests cover that chain:

- `checkpoints/manifest.json`: controller and exploratory-artifact identities;
- `data/training/manifest.json`: numeric offline-supervision identities;
- `data/evaluation_manifest.json`: ordered sample IDs and reference hashes;
- `results/manifest.json`: paper configurations, record availability,
  expected metrics, completion-token aggregates, and controlled VLM latency
  references.

Run both validators before comparing a reproduction:

```bash
make check
make test
sot data validate --config experiments/paper/llama-3.1-8b.yaml
```

## Supported reproduction routes

### Frozen-checkpoint evaluation

This is the paper-result path. It loads the immutable `.pt` controller,
uses deterministic decoding, writes one record per example, and rebuilds each
aggregate from those records:

```bash
sot evaluate --config experiments/paper/llama-3.1-8b.yaml
```

Change only the config path to run another backbone. Model and dataset access
are external to this repository; see `data/README.md` for the input schema.

### Controller refitting

The five paper configurations also include their numeric offline trajectories.
No prompt text or generated response is needed to fit the 582 parameters:

```bash
sot train --config experiments/paper/llama-3.1-8b.yaml
sot checkpoint verify outputs/training/llama-3.1-8b/controller.pt
```

The fit is deterministic under the released seed and software stack. The
stopping heads reproduce the frozen training output numerically. Historical
text-controller fits did not serialize the process-wide random-generator state,
so a clean refit can differ slightly in gate coefficients while following the
same objective and data; exact paper predictions use the frozen-checkpoint
route above. The VLM paper protocol fits the stopping head while keeping
evidence retrieval fixed; its released stopping-head refit is exact.

The result manifest labels a run as `bundled` when its publication-aligned,
privacy-safe per-example outputs are included and as `aggregate-only` when only
the frozen aggregate is available locally. Aggregate-only rows are comparison
targets, not reconstructed records; rerunning the corresponding YAML
configuration creates a new complete record set.

## Paper configurations

| Configuration | Backbone | Precision | Evaluation datasets |
|---|---|---:|---:|
| `llama-3.1-8b.yaml` | Meta-Llama-3.1-8B-Instruct | BF16 | 16 |
| `qwen2.5-14b.yaml` | Qwen2.5-14B-Instruct | BF16 | 16 |
| `mixtral-8x7b.yaml` | Mixtral-8x7B-Instruct-v0.1 | BF16 | 16 |
| `qwen2.5-vl-7b.yaml` | Qwen2.5-VL-7B-Instruct | BF16 | 3 |
| `qwen2.5-vl-32b-bf16.yaml` | Qwen2.5-VL-32B-Instruct | BF16 | 3 |

The text matrix uses GSM8K (500), MATH (400), DROP (300), FOLIO (203),
ProofWriter (400), BBH temporal sequences (100), HumanEval (164), MBPP (250),
CommonsenseQA (400), StrategyQA (400), BoolQ (400), MMLU (500), RACE (300),
HotpotQA (300), NarrativeQA (250), and MultiFieldQA from LongBench (110).

The VLM matrix uses the same ordered samples at both scales: A-OKVQA (200),
AI2D (150), and M3CoT (150). Images are loaded from local paths in the JSONL
records and are closed immediately after each example.

## Exploratory variants

The training-free and sentence-embedding variants reuse the text matrices:

```bash
sot evaluate --config experiments/exploratory/llama-3.1-8b-training-free.yaml
sot evaluate --config experiments/exploratory/llama-3.1-8b-embedding.yaml
```

Equivalent configs are provided for Qwen2.5-14B and Mixtral-8x7B. The
embedding configs explicitly load their PCA model and frozen threshold policy;
they require `OPENAI_API_KEY` and never persist the key.

The trajectory-quality experiment uses the three released judge artifacts:

```bash
sot judge evaluate \
  --checkpoint checkpoints/exploratory/judge/openai.joblib \
  --records data/judge/openai-test.jsonl \
  --output outputs/judge/openai.predictions.jsonl
```

To refit the paper-aligned PCA-32/MLP classifier from labeled trajectories:

```bash
sot judge train \
  --records data/judge/openai-train.jsonl \
  --output outputs/judge/openai.joblib \
  --embedding-model text-embedding-3-large \
  --pca-dim 32 --seed 42
```

## Metrics and token accounting

- Arithmetic and classification tasks use the task-specific canonical answer
  parser encoded in `sot.grading`.
- DROP, HotpotQA, NarrativeQA, and MultiFieldQA use token-level F1 where stated
  in the YAML config.
- HumanEval and MBPP use executable pass@1. Execution is disabled by default;
  set `SOT_ALLOW_UNSAFE_CODE_EVAL=1` only inside an isolated sandbox.
- `aggregated_completion_tokens` is the sum of observe-step completions and the
  final answer-commit completion. Prompt/context tokens are not counted as
  generated tokens.
- Each dataset summary is computed only after the number of completed records
  equals the configured sample count.

## Latency protocol

Wall-clock latency is hardware and load dependent and is not expected to match
across unrelated systems. For a controlled comparison, use one otherwise-idle
GPU, one evaluation process, the precision in the YAML config, identical
ordered samples, CUDA synchronization immediately before and after each
example, and the same software versions. Warm model loading and compilation
before retaining measurements. Report the mean of the per-example
`wall_seconds` field; do not combine timings collected while other processes
share the device.

The paper's VLM reference timings were collected on one dedicated 80GB A100.
For every model--dataset--method cell, the first ordered example was an
untimed warm-up and the following 20 ordered examples were timed. The SoT
means are recorded in `results/manifest.json`; these values are
hardware-specific checks rather than cross-system pass/fail thresholds.

## Environment

`requirements-tested.txt` records the package versions used by the release
checks. CUDA, driver, and PyTorch builds must be mutually compatible. The
runtime does not rely on machine-specific paths or implicit device/threshold
environment variables: device, precision, quantization, attention backend,
checkpoint, and any exploratory threshold policy are explicit in YAML.

## Integrity and privacy

`scripts/check_release.py` fails closed on checkpoint or training-data hash
mismatches, missing files, credentials, private filesystem roots, cloud
addresses, and private workflow metadata. Public result records contain
predictions, metrics, token counts, and reference hashes; they do not expose
dataset reference text. The release contains no API keys, local caches, model
weights, or private experiment paths.
