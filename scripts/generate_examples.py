#!/usr/bin/env python3
"""Generate the eight bundled StatePlay examples with one model load."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stateplay import NEG_PROMPT, StatePlayPipeline  # noqa: E402
from stateplay.utils import save_mp4  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate all eight bundled StatePlay examples")
    parser.add_argument("--base-model", default=str(REPO_ROOT / "base_model"), help="Base-model root containing Wan-AI/ (default: <repo>/base_model)")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "examples" / "checkpoint" / "StatePlay.safetensors"))
    parser.add_argument("--manifest", default=str(REPO_ROOT / "examples" / "manifest.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "examples" / "generated"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", metavar="ID", help="Optional IDs, e.g. --only 01 03. Default: all eight.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    examples_root = manifest_path.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.only:
        wanted = {item.zfill(2) for item in args.only}
        rows = [row for row in rows if row["id"].zfill(2) in wanted]
        missing = wanted - {row["id"].zfill(2) for row in rows}
        if missing:
            raise ValueError(f"Unknown example IDs: {sorted(missing)}")
    if not rows:
        raise ValueError("No examples selected")

    print(f"Loading StatePlay once for {len(rows)} example(s)...", flush=True)
    pipeline = StatePlayPipeline.from_pretrained(base_model_dir=args.base_model, checkpoint_path=args.checkpoint, torch_dtype=torch.bfloat16).to(args.device)

    run_records = []
    for position, row in enumerate(rows, start=1):
        input_dir = examples_root / row["input_dir"]
        video_output = output_dir / row["output_name"]
        state_output = video_output.with_name(f"{video_output.stem}_state.txt")
        if args.skip_existing and video_output.exists() and state_output.exists():
            print(f"[{position}/{len(rows)}] skip existing {video_output.name}", flush=True)
            continue

        prompt = (input_dir / "prompt.txt").read_text(encoding="utf-8").strip()
        actions = input_dir / "actions.parquet"
        with Image.open(input_dir / "first_frame.png") as image:
            result = pipeline(
                image=image.convert("RGB"), actions_parquet=str(actions), state_parquet=str(actions),
                state_txt=str(state_output), state_sampling="end", prompt=prompt, negative_prompt=NEG_PROMPT,
                num_frames=101, num_inference_steps=30, cfg_scale=5.0, state_cfg_scale=1.0,
                action_cfg_scale=1.0, seed=2,
            )
        save_mp4(result.frames[0], str(video_output), fps=20)
        run_records.append({
            "id": row["id"], "input_dir": row["input_dir"], "output_video": video_output.name,
            "output_state": state_output.name, "checkpoint": Path(args.checkpoint).name,
            "parameters": {"state_sampling": "end", "steps": 30, "cfg": 5.0, "state_cfg": 1.0,
                           "action_cfg": 1.0, "seed": 2, "num_frames": 101, "fps": 20},
        })
        print(f"[{position}/{len(rows)}] saved {video_output}", flush=True)

    (output_dir / "run_manifest.json").write_text(json.dumps(run_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Finished. Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
