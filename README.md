<p align="center">
  <img src="assets/hero.svg" alt="State-of-Thought: endogenous reasoning with lightweight state control" width="100%">
</p>

<h1 align="center">State-of-Thought</h1>

<p align="center">
  <strong>Endogenous reasoning over frozen language and vision-language backbones</strong><br>
  A compact dynamics-geometric state controls evidence access and decides when reasoning is ready to commit.
</p>

<table>
  <tr>
    <td align="center">
      <a href="https://arxiv.org/abs/2609.16055"><strong>📄 Read the paper</strong></a><br>
      <sub>Method, analysis, and complete results</sub>
    </td>
    <td align="center">
      <a href="https://gongzhiren.github.io/SoT-website/"><strong>🌐 Explore the project</strong></a><br>
      <sub>Visual overview of State-of-Thought</sub>
    </td>
    <td align="center">
      <a href="https://gongzhiren.github.io/SoT-website/tutorial.html"><strong>▶ Follow the tutorial</strong></a><br>
      <sub>A guided walkthrough of the method</sub>
    </td>
    <td align="center">
      <a href="https://github.com/GongZhiren/State-of-Thought/releases/tag/v1.0.0"><strong>📦 Use the release</strong></a><br>
      <sub>Code, checkpoints, and result records</sub>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://arxiv.org/abs/2609.16055"><img src="https://img.shields.io/badge/arXiv-2609.16055-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://github.com/GongZhiren/State-of-Thought/releases/tag/v1.0.0"><img src="https://img.shields.io/github/v/release/GongZhiren/State-of-Thought?display_name=tag&style=flat-square&color=2563eb" alt="GitHub release"></a>
  <a href="https://github.com/GongZhiren/State-of-Thought/actions/workflows/release-check.yml"><img src="https://img.shields.io/github/actions/workflow/status/GongZhiren/State-of-Thought/release-check.yml?branch=main&style=flat-square&label=release%20checks" alt="Release checks"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-16a34a?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#why-state-of-thought"><strong>Why SoT?</strong></a> ·
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#whats-in-the-release"><strong>Release contents</strong></a> ·
  <a href="#released-model-matrix"><strong>Model matrix</strong></a> ·
  <a href="REPRODUCIBILITY.md"><strong>Reproduce</strong></a> ·
  <a href="#paper"><strong>Citation</strong></a>
</p>

<p align="center">
  <sub>Zhiren Gong · Yikun Hou · Zihao Zeng · Ming Xiao · Chau Yuen · Wei Yang Bryan Lim</sub>
</p>

| **4D** reasoning state | **582** trainable parameters | **Frozen** backbones | **3 LLM + 2 VLM** configurations | **19** reasoning datasets |
|:---:|:---:|:---:|:---:|:---:|
| dynamics + geometry | evidence gate + stop head | no backbone updates | text and multimodal | publication-aligned protocols |

## Why State-of-Thought?

Most reasoning pipelines control a model from the outside—through repeated
sampling, search, fixed heuristics, or an additional planner. SoT instead reads
the model's evolving internal information transfer and turns it into an
explicit, inspectable reasoning state.

> **Core idea.** Reasoning control should follow the trajectory already forming
> inside the backbone: retrieve prior evidence when it becomes useful, and stop
> when the internal state indicates that the answer is ready.

<p align="center">
  <img src="assets/overview.png" alt="Comparison of external reasoning control and the State-of-Thought framework" width="92%">
</p>
<p align="center"><sub>SoT replaces externally imposed control with lightweight state-conditioned evidence selection and commitment.</sub></p>

At every reasoning step, SoT extracts four dynamics-geometric coordinates:

| Coordinate | What it captures | Control signal |
|---|---|---|
| **Dispersion** | how distributed the current representation is | breadth of the active reasoning state |
| **Velocity** | displacement across adjacent steps | how quickly reasoning is evolving |
| **Directional consistency** | alignment of successive displacements | whether the trajectory is stabilizing |
| **Instability** | local uncertainty in state evolution | whether more reasoning is still useful |

The resulting state drives a **16→32→1 evidence gate** (577 parameters) and a
**four-weight stopping head with one bias** (5 parameters). The backbone stays
frozen throughout.

## Quick start

### 1. Install

```bash
git clone https://github.com/GongZhiren/State-of-Thought.git
cd State-of-Thought
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Add `.[vlm]` for vision-language inputs or `.[judge]` for the trajectory-quality
experiment. Backbone weights are downloaded separately from Hugging Face;
gated models require accepting their licenses and running `hf auth login`.

### 2. Choose a reproduction path

| Goal | What it requires | Start here |
|---|---|---|
| Audit a released controller | CPU, seconds; no backbone | `sot checkpoint verify checkpoints/llama-3.1-8b/controller.pt` |
| Validate an evaluation split | Prepared data; no model inference | `sot data validate --config experiments/paper/llama-3.1-8b.yaml` |
| Evaluate the frozen controller | The named backbone and evaluation data | `sot evaluate --config experiments/paper/llama-3.1-8b.yaml` |
| Refit all 582 parameters | Released offline trajectories; no backbone forward pass | `sot train --config experiments/paper/llama-3.1-8b.yaml` |

Prepare evaluation data as described in [`data/README.md`](data/README.md).
A typical validation and evaluation flow is:

```bash
sot data validate --config experiments/paper/llama-3.1-8b.yaml
sot evaluate --config experiments/paper/llama-3.1-8b.yaml
```

To reproduce controller fitting from the released trajectories:

```bash
sot train --config experiments/paper/llama-3.1-8b.yaml
```

Runs are record-first and resumable. Every summary retains the configuration,
checkpoint identity, record hashes, token counts, and wall-clock measurements.

## What's in the release

| Component | Included | Start here |
|---|---|---|
| **Runtime** | text and vision-language state extraction, evidence selection, stopping, and evaluation | [`src/sot/`](src/sot/) |
| **Training** | end-to-end fitting from numeric offline reasoning trajectories | [`sot.training`](src/sot/training.py) |
| **Checkpoints** | five frozen main controllers plus training-free, embedding, and judge artifacts | [`checkpoints/`](checkpoints/) |
| **Experiments** | paper-aligned configurations for three LLM and two VLM backbones | [`experiments/paper/`](experiments/paper/) |
| **Results** | curated summaries, immutable manifests, and publication-aligned per-example records | [`results/`](results/) |
| **Data contract** | ordered sample/reference hashes and reproducible VLM preparation | [`data/`](data/) |
| **Verification** | checkpoint, result, privacy, lint, package, and unit-test gates | [`Makefile`](Makefile) |

### Released model matrix

| Modality | Released backbone configurations | Evaluation coverage |
|---|---|---|
| **Text** | Llama-3.1-8B, Qwen2.5-14B, Mixtral-8×7B | 16 fixed reasoning datasets per backbone |
| **Vision–language** | Qwen2.5-VL-7B, Qwen2.5-VL-32B (BF16) | A-OKVQA, AI2D, and M3CoT |

Curated metrics live in [`results/summaries/`](results/summaries/). Bundled
per-example records are under [`results/records/`](results/records/), and
[`results/manifest.json`](results/manifest.json) binds their provenance.

Training logs, temporary sweeps, failed or superseded runs, and unstructured
process files are intentionally excluded. Baseline implementations,
visualization-only analyses, interpretability probes, and defensive diagnostics
are outside the runtime release. Dataset and backbone weights remain governed
by their original licenses and are not redistributed.

## Reproducibility

The complete reproduction contract documents:

- exact model × dataset matrices and paper configurations;
- frozen checkpoint identities and SHA-256 hashes;
- ordered evaluation subsets and grading behavior;
- token accounting and latency protocols;
- training-free, embedding-based, and SoT-Judge extensions.

See **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** for the full workflow, or run
the repository-wide release gate directly:

```bash
make check test
```

## Paper

**[State of Thought Enables Endogenous Reasoning](https://arxiv.org/abs/2609.16055)**<br>
Zhiren Gong, Yikun Hou, Zihao Zeng, Ming Xiao, Chau Yuen, and Wei Yang Bryan Lim.

<details>
<summary><strong>BibTeX</strong></summary>

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

</details>

## License

The source code is released under the [MIT License](LICENSE). Model checkpoints,
datasets, and third-party software retain their respective licenses.

<p align="center"><sub>Built for transparent, end-to-end reproduction of State-of-Thought.</sub></p>
