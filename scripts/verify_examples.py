#!/usr/bin/env python3
"""Compare generated examples with the bundled reference videos."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_equal(left: Path, right: Path) -> tuple[bool, int, int]:
    left_iter = iio.imiter(left, plugin="FFMPEG")
    right_iter = iio.imiter(right, plugin="FFMPEG")
    frame_count = 0
    max_delta = 0
    sentinel = object()
    while True:
        left_frame = next(left_iter, sentinel)
        right_frame = next(right_iter, sentinel)
        if left_frame is sentinel or right_frame is sentinel:
            return left_frame is sentinel and right_frame is sentinel and max_delta == 0, frame_count, max_delta
        if left_frame.shape != right_frame.shape:
            return False, frame_count, 255
        delta = int(np.abs(left_frame.astype(np.int16) - right_frame.astype(np.int16)).max())
        max_delta = max(max_delta, delta)
        frame_count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify eight StatePlay example videos")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "examples" / "manifest.csv"))
    parser.add_argument("--generated-dir", default=str(REPO_ROOT / "examples" / "generated"))
    parser.add_argument("--reference-dir", default=str(REPO_ROOT / "examples" / "reference_outputs"))
    args = parser.parse_args()

    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failed = False
    for row in rows:
        generated = Path(args.generated_dir) / row["output_name"]
        reference = Path(args.reference_dir) / row["reference_name"]
        if not generated.is_file() or not reference.is_file():
            print(f"FAIL {row['id']}: missing {generated if not generated.is_file() else reference}")
            failed = True
            continue
        generated_hash, reference_hash = sha256(generated), sha256(reference)
        if generated_hash == reference_hash:
            print(f"PASS {row['id']}: byte-identical sha256={generated_hash}")
            continue
        equal, frame_count, max_delta = decoded_equal(generated, reference)
        if equal:
            print(f"PASS {row['id']}: decoded pixels identical ({frame_count} frames); container bytes differ")
        else:
            print(f"FAIL {row['id']}: generated={generated_hash} reference={reference_hash} frames_checked={frame_count} max_pixel_delta={max_delta}")
            failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
