# Publication-aligned results

This directory is the curated result release for **State of Thought Enables
Endogenous Reasoning**. It contains only the final, publication-aligned result
artifacts: compact human-readable summaries, an immutable machine-readable
manifest, and the per-example records that support the bundled aggregates.

Training logs, hyperparameter sweeps, interrupted runs, debug traces, temporary
exports, and superseded checkpoints are intentionally excluded.

## Layout

```text
results/
├── README.md
├── manifest.json
├── summaries/
│   ├── main_llm_accuracy.csv
│   ├── main_vlm.csv
│   └── exploratory.csv
├── schema/
│   └── prediction-record.schema.json
└── records/
    ├── main/
    │   ├── llm/llama-3.1-8b/
    │   └── vlm/{qwen2.5-vl-7b,qwen2.5-vl-32b-bf16}/
    └── exploratory/
        ├── llama-3.1-8b-training-free/
        └── llama-3.1-8b-embedding/
```

The CSV files are presentation-ready indexes rounded as in the paper. Exact
floating-point values, configuration paths, checkpoint hashes, record counts,
and record hashes live in [`manifest.json`](manifest.json), which is the
machine-readable source of truth.

## Released coverage

| Result family | Models / variants | Tasks | Public record status |
|---|---|---:|---|
| Main LLM | Llama-3.1-8B | 16 | Per-example records bundled |
| Main LLM | Qwen2.5-14B, Mixtral-8x7B | 16 each | Frozen aggregates; executable checkpoints/configs bundled |
| Main VLM | Qwen2.5-VL-7B, Qwen2.5-VL-32B BF16 | 3 each | Per-example records bundled |
| Limited-access extensions | Training-free SoT, SoT-Embedding on Llama-3.1-8B | LongBench | Per-example records bundled |

`record_status: "bundled"` means that the corresponding privacy-safe JSONL is
included and hash-locked. `record_status: "aggregate-only"` means that the
publication aggregate is preserved while no canonical per-example export is
claimed; the released checkpoint, configuration, split identity, and grader
remain available for a fresh evaluation.

## Record format

Each JSONL line is one immutable prediction record. The schema is documented in
[`schema/prediction-record.schema.json`](schema/prediction-record.schema.json).
The essential fields are:

- `id`, `dataset`, and `task_type`: identity of the evaluated item and task;
- `prediction_raw`: exact decoded answer retained for reproducible parsing;
- `prediction`: normalized answer consumed by the public grader;
- `reference_sha256`: reference identity without redistributing benchmark text;
- `metric` and `metric_value`: per-example contribution to the reported score;
- `completion_tokens`: observe-step plus final answer-commit tokens;
- `stopped_by`: optional terminal condition when retained by the source run.

Incorrect predictions are retained deliberately: these files are complete
evaluation records, not selected demonstrations. Prompts, gold-answer text,
private paths, credentials, process metadata, and intermediate reasoning logs
are not included.

## Verify and reproduce

Verify all bundled record hashes, counts, metrics, token aggregates, checkpoint
identities, and generated CSV summaries:

```bash
make check
```

Regenerate the human-readable summaries from the manifest:

```bash
python scripts/build_result_summaries.py
```

Re-run a paper configuration from its released controller checkpoint:

```bash
sot data validate --config experiments/paper/llama-3.1-8b.yaml
sot evaluate --config experiments/paper/llama-3.1-8b.yaml
```

New evaluations are written under `outputs/`, never into this frozen directory.
See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the full checkpoint,
dataset, grading, and latency protocol.
