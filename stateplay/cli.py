"""Command-line inference for StatePlay."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from . import NEG_PROMPT, StatePlayPipeline
from .utils import save_mp4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an SF3 rollout with StatePlay")
    parser.add_argument("--checkpoint", required=True, help="StatePlay .safetensors checkpoint")
    parser.add_argument("--base-model", required=True, help="Directory containing the Wan base assets")
    parser.add_argument("--image", required=True, help="First-frame PNG/JPG")
    parser.add_argument("--actions", required=True, help="Action/state parquet")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="Positive prompt text")
    prompt.add_argument("--prompt-file", help="UTF-8 text file containing the positive prompt")
    parser.add_argument("--output", default="output.mp4", help="Generated MP4 path")
    parser.add_argument("--state-output", default=None, help="Predicted, ground-truth, and error state text path")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--state-cfg", type=float, default=1.0)
    parser.add_argument("--action-cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=101)
    parser.add_argument("--state-sampling", choices=("interpolation", "avg", "end"), default="end")
    parser.add_argument("--negative-prompt", default=NEG_PROMPT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.prompt_file:
        positive_prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        positive_prompt = args.prompt

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_output = Path(args.state_output) if args.state_output else output.with_name(f"{output.stem}_state.txt")
    state_output.parent.mkdir(parents=True, exist_ok=True)

    pipeline = StatePlayPipeline.from_pretrained(
        base_model_dir=args.base_model,
        checkpoint_path=args.checkpoint,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    result = pipeline(
        image=Image.open(args.image),
        actions_parquet=args.actions,
        state_parquet=args.actions,
        state_txt=str(state_output),
        state_sampling=args.state_sampling,
        prompt=positive_prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        state_cfg_scale=args.state_cfg,
        action_cfg_scale=args.action_cfg,
        seed=args.seed,
    )
    save_mp4(result.frames[0], str(output), fps=20)
    print(f"Saved video: {output.resolve()}")
    print(f"Saved state: {state_output.resolve()}")


if __name__ == "__main__":
    main()
