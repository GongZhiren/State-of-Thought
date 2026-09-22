# Controller checkpoints

Each released controller is provided in both forms:

- `controller.json`: human-readable controller/config artifact.
- `controller.pt`: tensor-native checkpoint generated from that exact JSON.

Each pair is linked in `manifest.json` by SHA-256. `sot checkpoint verify`
checks the embedded source hash, tensor shapes, finite values, and exact
round-trip equality before a checkpoint can be used for evaluation.

Backbone model weights are not included. Download them under their original
licenses and provide the local path or Hugging Face identifier in the experiment
configuration.

## Exploratory artifacts

`exploratory/training-free/` contains fixed, zero-learned-parameter policies;
the main controller checkpoint supplies only the shared normalization and
inference configuration for these runs. `exploratory/embed/` contains the
controller, PCA projection, and frozen thresholds used by the sentence-
embedding variant. Its fifth compatibility stop coefficient is omitted in `.pt`
because the reported runtime consumed the first four state coordinates.

`exploratory/judge/` contains the three fitted post-hoc trajectory classifiers.
Their estimators and PCA transforms are unchanged from the source experiment
artifacts; machine-local training paths were removed. The release tests check
their hashes, dimensions, and fixed-probe predictions.
