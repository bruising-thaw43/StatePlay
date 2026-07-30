# StatePlay

[**Project Page**](https://jimntu.github.io/stateplay_page/) ·
[**Paper**](https://arxiv.org/abs/2607.26754) ·
[**Code**](.) ·
[**Dataset**](https://huggingface.co/datasets/onepiece1999/StatePlay-Dataset) ·
[**Model**](https://huggingface.co/onepiece1999/StatePlay)

StatePlay is a state-aware game world model for mechanics-consistent generation in *Street Fighter III*. Unlike pixel-only game world models, StatePlay jointly predicts future visual frames and explicit internal game states—including the timer, player and opponent health points, and both skill meters—conditioned on an initial frame, a text prompt, and player actions.

StatePlay uses a Mixture-of-Transformers (MoT)-style architecture with specialized visual and state branches connected through shared joint attention. This design allows predicted state dynamics to guide frame generation while preserving modality-specific representations. The released `StatePlay.safetensors` checkpoint contains the complete StatePlay DiT state dictionary; this repository provides the corresponding inference API, training path, and reproducible examples.

The repository contains:

- a standalone inference package (`stateplay/`);
- the exact StatePlay training path and its minimal DiffSynth dependencies (`training/`, `diffsynth/`);
- the expected location and download instructions for the 50,000-step StatePlay checkpoint;
- eight self-contained example inputs;
- one-command training, single-example inference, and eight-example generation.

Model weights and example output videos are not stored in this Git repository. Download the weights separately and generate the videos locally using the commands below.

## Repository layout

```text
StatePlay/
├── stateplay/                  # public inference API and CLI
├── training/                   # StatePlay training entry point and SF3 data logic
├── diffsynth/                  # minimal training-only Wan/StatePlay dependency subset
├── scripts/
│   ├── inference.sh            # generate one video
│   ├── run_examples.sh         # generate all eight examples
│   ├── generate_examples.py    # one model load, eight generations
│   ├── verify_examples.py      # optional comparison against local references
│   └── train.sh                # exact StatePlay training configuration
└── examples/
    ├── checkpoint/             # local checkpoint; ignored by Git
    ├── inputs/                 # eight first frames, prompts, and action/state parquets
    ├── reference_outputs/      # optional local references; ignored by Git
    ├── generated/              # locally generated demos; ignored by Git
    └── manifest.csv
```

Unrelated DiffSynth architectures and pipelines are intentionally not included.

## Installation

Python 3.10 or newer and a CUDA GPU are required. The reference runs use bfloat16.

```bash
git clone https://github.com/Jimntu/StatePlay.git
cd StatePlay
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Download model weights

StatePlay does not download weights automatically at runtime. Download the following resources from Hugging Face and keep the directory structure shown below.

> [!NOTE]
> The StatePlay checkpoint is available at [`onepiece1999/StatePlay`](https://huggingface.co/onepiece1999/StatePlay).

| Resource | Required for | Hugging Face repository | Files to download |
|---|---|---|---|
| StatePlay checkpoint | inference | [`onepiece1999/StatePlay`](https://huggingface.co/onepiece1999/StatePlay) | `StatePlay.safetensors` |
| Wan2.2 VAE and T5 | inference and training | [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth` |
| UMT5 tokenizer | inference and training | [`Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main/google/umt5-xxl) | everything under `google/umt5-xxl/` |
| Wan2.2 DiT base weights | training only | [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | the three `diffusion_pytorch_model-*.safetensors` shards |

The StatePlay checkpoint is a complete DiT state dict, so the three Wan2.2 diffusion shards are **not needed for inference**. They are only used to initialize the DiT when training StatePlay.

### Download the inference weights

Install the Hugging Face CLI, then run these commands from the repository root:

```bash
python -m pip install -U huggingface_hub

hf download onepiece1999/StatePlay \
  StatePlay.safetensors \
  --local-dir examples/checkpoint

hf download Wan-AI/Wan2.2-TI2V-5B \
  Wan2.2_VAE.pth \
  models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir base_model/Wan-AI/Wan2.2-TI2V-5B

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local-dir base_model/Wan-AI/Wan2.1-T2V-1.3B
```

After downloading, the inference assets must have this layout:

```text
StatePlay/
├── examples/
│   └── checkpoint/
│       └── StatePlay.safetensors
└── base_model/
    └── Wan-AI/
        ├── Wan2.2-TI2V-5B/
        │   ├── Wan2.2_VAE.pth
        │   └── models_t5_umt5-xxl-enc-bf16.pth
        └── Wan2.1-T2V-1.3B/
            └── google/
                └── umt5-xxl/
                    ├── special_tokens_map.json
                    ├── spiece.model
                    ├── tokenizer.json
                    └── tokenizer_config.json
```

The expected StatePlay checkpoint path and checksum are:

```text
examples/checkpoint/StatePlay.safetensors
SHA256 aba9747fcd3f0c0a3546941253cd99ecd5dbc2c69747d647b91de84ee41f722f
```

The checkpoint is about 11.5 GB and is distributed separately rather than as an ordinary Git object.

### Download the additional training weights

Training also requires the Wan2.2 DiT base weights. Download the three shards into the same Wan2.2 directory:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  diffusion_pytorch_model-00001-of-00003.safetensors \
  diffusion_pytorch_model-00002-of-00003.safetensors \
  diffusion_pytorch_model-00003-of-00003.safetensors \
  --local-dir base_model/Wan-AI/Wan2.2-TI2V-5B
```

The complete training directory is therefore:

```text
base_model/
└── Wan-AI/
    ├── Wan2.2-TI2V-5B/
    │   ├── diffusion_pytorch_model-00001-of-00003.safetensors
    │   ├── diffusion_pytorch_model-00002-of-00003.safetensors
    │   ├── diffusion_pytorch_model-00003-of-00003.safetensors
    │   ├── models_t5_umt5-xxl-enc-bf16.pth
    │   └── Wan2.2_VAE.pth
    └── Wan2.1-T2V-1.3B/
        └── google/umt5-xxl/
            └── ... tokenizer files ...
```

### Use a custom model location

All scripts use `./base_model` and `./examples/checkpoint/StatePlay.safetensors` by default. To store the weights elsewhere, point the environment variables to the corresponding locations:

```bash
export STATEPLAY_BASE_MODEL=/absolute/path/to/base_model
export STATEPLAY_CHECKPOINT=/absolute/path/to/StatePlay.safetensors
```

`STATEPLAY_BASE_MODEL` must point to the directory that directly contains `Wan-AI/`, not to `Wan-AI/` or `Wan2.2-TI2V-5B/`.

## Generate the eight examples

```bash
export STATEPLAY_BASE_MODEL=/path/to/base_model
export CUDA_VISIBLE_DEVICES=0
./scripts/run_examples.sh
```

The model is loaded once, then all eight videos are generated into `examples/generated/`. Both `examples/generated/` and `examples/reference_outputs/` are intentionally ignored by Git, so generated and reference media remain local.

The eight bundled inputs are:

| ID | Distribution |
|---:|---|
| 01 | macro success |
| 02 | macro success |
| 03 | result win |
| 04 | result win |
| 05 | result lose |
| 06 | result lose |
| 07 | normal |
| 08 | normal |

Each input directory contains `first_frame.png`, `prompt.txt`, `actions.parquet`, a human-readable `actions_readable.csv`, and `metadata.json`.

The generation settings are fixed to the original demos:

| Setting | Value |
|---|---:|
| checkpoint | StatePlay (step 50,000) |
| resolution | 480 × 832 |
| frames / fps | 101 / 20 |
| inference steps | 30 |
| text CFG | 5.0 |
| state CFG | 1.0 |
| action CFG | 1.0 |
| state sampling | `end` |
| seed | 2 |

To invoke the Python generator directly:

```bash
python scripts/generate_examples.py \
  --base-model "$STATEPLAY_BASE_MODEL" \
  --checkpoint examples/checkpoint/StatePlay.safetensors
```

The generator reads `examples/manifest.csv`. For each row it loads the corresponding `first_frame.png`, `prompt.txt`, and `actions.parquet` from `examples/inputs/`, runs StatePlay with the fixed settings above, and writes an MP4 plus a predicted-state TXT file to `examples/generated/`. It also writes `examples/generated/run_manifest.json` with the checkpoint name and generation parameters.

If you separately have local reference videos, place them in `examples/reference_outputs/` using the filenames in `examples/manifest.csv`, then optionally compare them:

```bash
python scripts/verify_examples.py
```

The verifier first checks full-file SHA256. If container bytes differ, it decodes both videos and compares every RGB pixel. Reference videos are not required for generation and are not distributed through this Git repository.

## Generate one video

```bash
export STATEPLAY_BASE_MODEL=/path/to/base_model
export CUDA_VISIBLE_DEVICES=0

./scripts/inference.sh \
  --image examples/inputs/01_macro_success_clip/first_frame.png \
  --actions examples/inputs/01_macro_success_clip/actions.parquet \
  --prompt-file examples/inputs/01_macro_success_clip/prompt.txt \
  --output output.mp4
```

The wrapper supplies the default checkpoint and base-model paths. Override them with `STATEPLAY_CHECKPOINT` and `STATEPLAY_BASE_MODEL`. Run `./scripts/inference.sh --help` for optional parameters. The command also writes predicted state values beside the video as `output_state.txt`.

State output is reported at the model's native latent rate: 26 rows for a 101-frame video. The parquet's full state sequence is loaded and downsampled with the same `end` rule for alignment. Latent row 0 is the conditioned initial-state token; rows 1–25 are predictions. Each TXT row contains `*_pred`, `*_true`, and `*_err` side by side.

Python API:

```python
import torch
from PIL import Image
from stateplay import StatePlayPipeline

pipe = StatePlayPipeline.from_pretrained(
    base_model_dir="/path/to/base_model",
    checkpoint_path="examples/checkpoint/StatePlay.safetensors",
    torch_dtype=torch.bfloat16,
).to("cuda")

result = pipe(
    image=Image.open("first_frame.png"),
    actions_parquet="actions.parquet",
    state_parquet="actions.parquet",
    prompt="SF3 Game.",
    state_sampling="end",
    num_frames=101,
    num_inference_steps=30,
    cfg_scale=5.0,
    state_cfg_scale=1.0,
    action_cfg_scale=1.0,
    seed=2,
)
```

## Train StatePlay

The training wrapper uses the StatePlay configuration: full DiT fine-tuning, 480×832, 101 frames, state sampling at the end of each latent-time bin, clean-state Smooth-L1 loss, video/state weights of 1.0, learning rate `5e-5`, prompt dropout `0.1`, no action dropout, and a 10-frame action hold window.

Expected dataset layout:

```text
$STATEPLAY_DATA_ROOT/
├── metadata_state_polish.csv
├── <video files referenced by CSV>
└── <action parquet files referenced by CSV>
```

The metadata CSV must include `video`, `action`, and `prompt`. Each parquet must include the SF3 button columns used by the profile and the state columns `timer`, `p1_hp`/`hp1`, `p2_hp`/`hp2`, `p1_meter`/`meter1`, and `p2_meter`/`meter2`.

Run on four GPUs:

```bash
export STATEPLAY_BASE_MODEL=/path/to/base_model
export STATEPLAY_DATA_ROOT=/path/to/SF3/train/clips_5s
export STATEPLAY_OUTPUT=/path/to/output/StatePlay
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/train.sh
```

Important overrides:

```bash
# Original H200 run reserved 100 GiB of each GPU's allocator cache.
# Use 0 on smaller GPUs or when this reservation is unnecessary.
RESERVE_CUDA_MEMORY_GB=0 ./scripts/train.sh

# Resume from a full DiT checkpoint (optimizer state is not restored).
./scripts/train.sh --resume_from_ckpt /path/to/step-N.safetensors

# Change checkpoint interval.
SAVE_STEPS=1000 ./scripts/train.sh
```

The wrapper derives the process count from `CUDA_VISIBLE_DEVICES`. All model and dataset paths are configurable; no private absolute path is embedded in the public scripts.

## Architecture naming

The repository exposes one architecture named `StatePlay`. There is no model-version selector in either training or inference.

## License and third-party assets

See [LICENSE](LICENSE) and [NOTICE](NOTICE). Wan base weights, the StatePlay checkpoint, SF3 data, and example media may be governed by separate terms; confirm redistribution rights before publishing those assets.
