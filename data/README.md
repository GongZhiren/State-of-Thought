# Data preparation

State-of-Thought does not redistribute third-party datasets. Download each
dataset from its official source under its original license, select the split
specified by the experiment configuration, and write one JSON object per line
to the configured path under `data/eval_fixed/` (text) or `data/vlm/` (vision-language).

The release includes `data/evaluation_manifest.json`, which records the exact
source split, deterministic subset rule, ordered sample IDs, and SHA-256 hashes
of reference answers without publishing the benchmark text. The VLM entries
also pin the exact Hugging Face dataset revision. After preparing a dataset,
validate it before evaluation:

```bash
sot data validate --config experiments/paper/llama-3.1-8b.yaml
sot data validate --config experiments/paper/qwen2.5-vl-7b.yaml
```

Validation fails on a missing or reordered sample, a changed reference answer,
an incorrect sample count, or a missing VLM image. This makes accidental split
drift visible before an expensive model run.

The VLM subsets can be materialized directly from the pinned sources:

```bash
pip install -e '.[data]'
python scripts/prepare_vlm_data.py
sot data validate --config experiments/paper/qwen2.5-vl-7b.yaml
```

## Evaluation records

Every record requires a question and reference answer:

```json
{"id":"stable-source-id","question":"...","answer":"..."}
```

Multiple-choice rows may additionally contain either `choices` or
`options`; vision-language rows contain `image_path` with a path relative to the
repository/experiment root (or an absolute path supplied locally). Task-specific fields are
preserved and passed to the grader. Keep source order fixed: the evaluator takes
the first `max_samples` rows and derives the same content ID used by the paper
run when `id` is absent. Repeated upstream IDs are supported and resume safely.

The paper configurations expect GSM8K, MATH, DROP, FOLIO, ProofWriter, BBH
temporal sequences, HumanEval, MBPP, CommonsenseQA, StrategyQA, BoolQ, MMLU,
RACE, HotpotQA, NarrativeQA, MultiFieldQA from LongBench, A-OKVQA, AI2D, and
M3CoT. Exact sample counts and task metrics are encoded in
`experiments/paper/*.yaml`.

The selected sources are GSM8K, MATH, DROP, FOLIO, ProofWriter (depth 0), BBH
Temporal Sequences, HumanEval, MBPP, CommonsenseQA, StrategyQA, BoolQ, MMLU,
RACE, HotpotQA, NarrativeQA, LongBench MultiFieldQA-en, A-OKVQA, AI2D, and
M3CoT. Text subsets are formed by stable sorting on the source ID and taking
the configured head count; the exact split and cap are stored per dataset in
the manifest. A-OKVQA, AI2D, and M3CoT use the pinned Hub revisions and their
manifested source-order prefixes. Preserve the released order rather than
sampling a new subset.

## Offline controller trajectories

The training entry point consumes
`data/training/<backbone>/offline_trajectories.jsonl.gz`.
Each line represents one reasoning step and contains:

```json
{
  "problem_id": "stable-id",
  "dataset": "gsm8k",
  "task_type": "math_qa",
  "split": "train",
  "traj_correct": true,
  "step_index": 3,
  "m_t": [0.1, -0.2, 0.3, 0.4, 0.0],
  "history": [{
    "m_j": [0.0, -0.1, 0.2, 0.3],
    "retrieval_trainable": 1,
    "gate_hard_label": 1,
    "gate_soft_label": 0.83
  }],
  "teacher_stop_score": 0.72,
  "stop_label": 1,
  "step_len_tokens": 31,
  "traj_quality": 1.0
}
```

The controller sees only numeric state/teacher fields; prompts and generated
text are not required by `sot train`. The release manifest records the SHA-256
identity of each paper training file. Training, validation, and evaluation IDs
must be disjoint.

## SoT-Judge trajectories

`sot judge train` and `sot judge evaluate` accept labeled generated
trajectories with either `sentence_units` or `prediction_raw`:

```json
{"id":"x1","dataset":"gsm8k","sentence_units":["Step one.","Therefore ..."],"correct":true}
```

Only the sentence text is sent to the configured embedding service. API keys
are read from the environment and are never written to checkpoints or results.
