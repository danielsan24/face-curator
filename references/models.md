# Models

| Name | File (in the models registry) | Input | How to register |
|---|---|---|---|
| SCRFD 10G detector | `buffalo_l/det_10g.onnx` | 640×640, RGB, (x-127.5)/128 | `models add-buffalo` |
| `buffalo_l` (reference) | `buffalo_l/w600k_r50.onnx` | 112×112 aligned, RGB, (x-127.5)/127.5 | `models add-buffalo` |
| `antelopev2` (challenger) | `antelopev2/glintr100.onnx` | same as buffalo_l | `models add antelopev2 --file PATH` |
| `adaface_ir101_webface12m` (challenger) | `adaface/adaface_ir101_webface12m.onnx` | 112×112 aligned, **BGR**, (x-127.5)/127.5 | `models add adaface_ir101_webface12m --file PATH` |

No weights are part of this repository. Inference runs on the CPU (onnxruntime's CPU provider).

## Getting buffalo_l

1. Download `buffalo_l.zip` (about 280 MB) from the InsightFace model zoo:
   <https://github.com/deepinsight/insightface/tree/master/model_zoo> (the "buffalo_l" pack; the Python package
   `insightface` downloads the same file on first use).
2. Unzip it into `~/.insightface/models/buffalo_l/`. Face Curator uses two of its files:

   | File | Size | sha256 |
   |---|---|---|
   | `det_10g.onnx` | about 17 MB (16,923,827 bytes) | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
   | `w600k_r50.onnx` | about 174 MB (174,383,860 bytes) | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |

3. `facecurator models add-buffalo` (or `--dir DIR` if the files are elsewhere).

Licence: InsightFace publishes its pretrained models (buffalo_l, antelopev2) for **non-commercial research use
only**: <https://github.com/deepinsight/insightface#license>. They are not covered by this repository's MIT
licence. AdaFace's weights are MIT.

## What registration checks

`models add-buffalo` hashes both files, refuses a file whose sha256 is not the one in the table above (the
hashes are pinned in `scripts/fc/embed.py`, `PINNED`), and records path, size and hash in
`<data_dir>/models/models.lock.json`. Every time a model is loaded, once per process, the file is hashed again
and refused if it no longer matches its registration; the two buffalo_l names must also still match the pinned
hashes, so editing the lock file does not get another file loaded under those names.

A challenger is a different model, registered by its own name: `models add NAME --file PATH`. There is no
pinned hash for it; the hash of the file you register is recorded and checked from then on. Download it only
from a source you trust: a model file is code-like input to the ONNX runtime.

AdaFace is published as a PyTorch checkpoint (`adaface_ir101_webface12m.ckpt`, github.com/mk-minchul/AdaFace).
It needs an ONNX export with a 1×3×112×112 input whose first output is the embedding. The channel-order trap is
real: the original code feeds BGR. A wrong order still produces plausible-looking scores, so evaluate before
trusting it.

## Choosing a challenger

1. Register it, set `models.challenger` in the config, and run `run --until assign`. Embeddings are computed for
   every face, which re-detects each photo to get landmarks: about 1 h for 40,000 photos.
2. Compare the two `calibration` rows of that run: the recall at the same 99.5% precision, and the 95% lower
   bound. Keep the challenger only if it is better on the sibling cards too.
3. With a challenger set, an acceptance by the reference model is vetoed when the challenger picks someone else.
   The embeddings are never averaged.

After changing a model file, onnxruntime or OpenCV, run `selftest`: it re-embeds stored faces from local images
and fails if the new embeddings are not comparable with the stored ones.
