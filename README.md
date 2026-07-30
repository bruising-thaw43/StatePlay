# StatePlay

[**Project Page**](https://jimntu.github.io/stateplay_page/) ·
[**Paper**](https://arxiv.org/abs/2607.26754) ·
[**Code**](https://github.com/Jimntu/StatePlay) ·
[**Dataset**](https://huggingface.co/datasets/onepiece1999/StatePlay-Dataset) ·
[**Model**](https://huggingface.co/onepiece1999/StatePlay)

![StatePlay mechanics-consistent generation](assets/teaser.webp)

## Abstract

Recent game world models can generate visually realistic and interactive
environments conditioned on player actions. However, games are not defined by
pixels alone; they are governed by explicit mechanics, namely state-dependent
rules that control health reduction, skill activation, and game termination.
These mechanics depend on precise internal states, such as health points, skill
meters, and timers, which are tightly coupled with visual observations and
determine how gameplay evolves. Without modeling these state dynamics, existing
game world models may generate visually plausible rollouts but violate the
underlying game rules. In this paper, we propose **StatePlay**, a novel
state-aware game world model that jointly predicts visual content and game
states to promote mechanics-consistent generation. StatePlay adopts a
mixture-of-transformers (MoT)-style architecture that preserves specialized
visual and state representations while enabling cross-modal interaction,
allowing predicted states to guide frame generation. Each branch is further
optimized with a distinct objective suited to its modality. Experiments show
that StatePlay achieves an average normalized L1 distance below 0.06 for state
prediction. Furthermore, compared with models without explicit state modeling,
our method improves mechanics fidelity in generated game rollouts by 18.6%.
Overall, our work highlights the importance of state-aware game world modeling
and advances beyond pixel-level realism toward complete and mechanically
faithful game generation.

![StatePlay architecture](https://jimntu.github.io/stateplay_page/static/images/stateplay.png)

## Installation

Requirements: Python 3.10+, CUDA, and a GPU with bfloat16 support.

```bash
git clone https://github.com/Jimntu/StatePlay.git
cd StatePlay
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Model downloads

| Model | Files used by StatePlay | Download |
|---|---|---|
| StatePlay | `StatePlay.safetensors` | [Hugging Face](https://huggingface.co/onepiece1999/StatePlay/tree/main) |
| Wan2.2-TI2V-5B | VAE, T5 encoder, and DiT initialization weights | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/main) |
| Wan2.1-T2V-1.3B | UMT5 tokenizer | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main) |

Expected inference layout:

```text
StatePlay/
├── examples/checkpoint/StatePlay.safetensors
└── base_model/Wan-AI/
    ├── Wan2.2-TI2V-5B/
    │   ├── Wan2.2_VAE.pth
    │   └── models_t5_umt5-xxl-enc-bf16.pth
    └── Wan2.1-T2V-1.3B/google/umt5-xxl/
        └── ... tokenizer files ...
```

To keep weights elsewhere:

```bash
export STATEPLAY_BASE_MODEL=/absolute/path/to/base_model
export STATEPLAY_CHECKPOINT=/absolute/path/to/StatePlay.safetensors
```

`STATEPLAY_BASE_MODEL` must directly contain `Wan-AI/`.

## Code architecture

StatePlay uses separate visual and state transformer branches with shared joint
attention. The visual branch predicts video latents; the state branch predicts
the five normalized game states. Both branches are conditioned on the initial
frame, text prompt, and action sequence.

```text
StatePlay/
├── stateplay/
│   ├── pipeline.py             # public inference pipeline
│   ├── cli.py                  # command-line inference
│   └── models/
│       ├── dit.py              # StatePlay visual/state DiT
│       ├── vae.py              # video VAE wrapper
│       └── text_encoder.py     # text encoder wrapper
├── training/
│   ├── train.py                # training entry point
│   ├── runner_with_state.py    # optimization/checkpoint loop
│   └── data/                   # SF3 action, state, and prompt loading
├── diffsynth/                  # minimal Wan/StatePlay dependencies
├── scripts/
│   ├── inference.sh
│   ├── run_examples.sh
│   └── train.sh
└── examples/
    ├── inputs/                 # eight bundled inputs
    ├── checkpoint/             # local checkpoint
    └── generated/              # generated videos and state predictions
```

## Run examples

Generate all eight bundled examples:

```bash
export CUDA_VISIBLE_DEVICES=0
./scripts/run_examples.sh
```

Generate selected examples:

```bash
./scripts/run_examples.sh --only 01 03
```

Outputs are written to `examples/generated/`. Each example produces an MP4 and
a `_state.txt` file. The model is loaded once for the entire run.

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0

./scripts/inference.sh \
  --image examples/inputs/01_macro_success_clip/first_frame.png \
  --actions examples/inputs/01_macro_success_clip/actions.parquet \
  --prompt-file examples/inputs/01_macro_success_clip/prompt.txt \
  --output output.mp4
```

This writes `output.mp4` and `output_state.txt`. Defaults are 101 frames,
30 denoising steps, text CFG 5.0, state/action CFG 1.0, and seed 2.

Use custom model paths through `STATEPLAY_BASE_MODEL` and
`STATEPLAY_CHECKPOINT`, or inspect all options with:

```bash
./scripts/inference.sh --help
```

## Training

Download the [StatePlay dataset](https://huggingface.co/datasets/onepiece1999/StatePlay-Dataset/tree/main)
and the Wan2.2 DiT initialization weights listed in the model table above.

Run training:

```bash
export STATEPLAY_BASE_MODEL="$PWD/base_model"
export STATEPLAY_DATA_ROOT="$PWD/data/StatePlay-Dataset/SF3"
export STATEPLAY_OUTPUT="$PWD/outputs/StatePlay"
export CUDA_VISIBLE_DEVICES=0,1,2,3

./scripts/train.sh
```

The script derives the process count from `CUDA_VISIBLE_DEVICES`. It trains at
480×832 with 101 frames, learning rate `5e-5`, state sampling `end`, and saves
every 500 steps. The released model is the checkpoint at **step 40,000**; stop
training after that checkpoint is written.

For GPUs that do not need the original H200 memory reservation:

```bash
RESERVE_CUDA_MEMORY_GB=0 ./scripts/train.sh
```

Resume from a DiT checkpoint:

```bash
./scripts/train.sh --resume_from_ckpt /path/to/step-N.safetensors
```
