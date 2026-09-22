<div align="center">

# State-of-Thought

### Endogenous reasoning over frozen language and vision-language backbones

**State-of-Thought (SoT) turns a compact four-dimensional internal state into
selective access to prior reasoning evidence and a learned stopping decision.
The controller has only 582 trainable parameters; the backbone remains frozen.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Release checks](https://github.com/GongZhiren/State-of-Thought/actions/workflows/release-check.yml/badge.svg)](https://github.com/GongZhiren/State-of-Thought/actions/workflows/release-check.yml)

[Paper](https://arxiv.org/abs/2609.16055) · [Project page](https://gongzhiren.github.io/SoT-website/) ·
[Tutorial](https://gongzhiren.github.io/SoT-website/tutorial.html) ·
[Quick start](#quick-start) · [Reproduce](REPRODUCIBILITY.md)

</div>

<p align="center">
  <img src="assets/overview.png" alt="State-of-Thought method overview" width="86%">
</p>

## Method

At each reasoning step, SoT extracts four dynamics-geometric coordinates from
the backbone's internal information transfer:

- dispersion, capturing how distributed the current representation is;
- velocity, measuring state displacement across adjacent steps;
- directional consistency, measuring whether successive displacements align;
- instability, capturing local uncertainty in the evolving state.

A small controller uses this state to activate historical reasoning units that
are useful now and to decide when the trajectory is ready to commit an answer.
The released controller contains a 16-to-32-to-1 evidence gate (577 parameters)
and a four-weight stopping head with one bias (5 parameters).

## Installation

```bash
git clone https://github.com/GongZhiren/State-of-Thought.git
cd State-of-Thought
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install `.[vlm]` for vision-language inputs and `.[judge]` for the
trajectory-quality experiment. Backbone weights are downloaded separately from
Hugging Face; gated models require accepting their licenses and authenticating
with `hf auth login`.

## Quick start

Validate a controller without loading a backbone:

```bash
sot checkpoint verify checkpoints/llama-3.1-8b/controller.pt
```

Place the evaluation JSONL files described in [`data/README.md`](data/README.md),
verify that their ordered samples, references, and VLM images match the released
paper split, then run a paper configuration:

```bash
sot data validate --config experiments/paper/llama-3.1-8b.yaml
sot evaluate --config experiments/paper/llama-3.1-8b.yaml
```

Runs are record-first and resumable. Each example is appended immediately, and
the final summary records the configuration, checkpoint, record hashes, token
counts, and wall-clock measurement.

To refit the lightweight controller from the released offline trajectories:

```bash
sot train --config experiments/paper/llama-3.1-8b.yaml
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for exact model/dataset matrices,
checkpoint identities, evaluation guarantees, and the training-free,
embedding-based, and SoT-Judge experiments.

## Released scope

This repository contains:

- end-to-end controller fitting from offline reasoning trajectories;
- text and vision-language checkpoint inference;
- the paper's three LLM and two VLM configurations;
- training-free, embedding-based, and trajectory-judge extensions;
- curated paper-result summaries, immutable manifests, and the recoverable
  publication-aligned per-example records.

The [`results/`](results/) release is organized separately from runtime output:
human-readable CSV summaries are kept at the top of the result bundle, while
the immutable JSONL records used to verify them live under a model/task
hierarchy. Training logs, intermediate sweeps, and failed or superseded runs
are not included.

Baseline implementations, visualization-only analyses, interpretability
probes, and defensive diagnostics are not needed to run or validate SoT and are
outside this release. Dataset and backbone weights remain governed by their
original licenses and are not redistributed here.

## Paper

**State of Thought Enables Endogenous Reasoning**
Zhiren Gong, Yikun Hou, Zihao Zeng, Ming Xiao, Chau Yuen, and Wei Yang Bryan Lim.

- Paper: <https://arxiv.org/abs/2609.16055>
- Project page: <https://gongzhiren.github.io/SoT-website/>
- Tutorial: <https://gongzhiren.github.io/SoT-website/tutorial.html>

## Citation

```bibtex
@misc{gong2026state,
  title         = {State of Thought Enables Endogenous Reasoning},
  author        = {Zhiren Gong and Yikun Hou and Zihao Zeng and Ming Xiao and Chau Yuen and Wei Yang Bryan Lim},
  year          = {2026},
  eprint        = {2609.16055},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.16055}
}
```

## License

The source code is released under the [MIT License](LICENSE). Model checkpoints,
datasets, and third-party software retain their respective licenses.
