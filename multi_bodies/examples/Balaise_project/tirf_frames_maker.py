#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def parse_config(path: str) -> Tuple[List[np.ndarray], int]:
    frames: List[np.ndarray] = []
    n_expected = None

    with open(path, "r", encoding="utf-8") as f:
        lines = (ln.strip() for ln in f)
        while True:
            for ln in lines:
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) == 1 and parts[0].lstrip("-").isdigit():
                    n = int(parts[0])
                    if n <= 0:
                        raise ValueError(f"Invalid particle count: {n}")
                    if n_expected is None:
                        n_expected = n
                    elif n_expected != n:
                        raise ValueError(
                            f"Inconsistent particle count: {n} (expected {n_expected})"
                        )
                    break
                else:
                    raise ValueError(f"Expected particle count line, got: {ln}")
            else:
                break

            xyz = []
            for _ in range(n):
                try:
                    ln = next(lines).strip()
                except StopIteration:
                    raise ValueError("Unexpected EOF inside a timestep.")
                vals = ln.split()
                if len(vals) < 3:
                    raise ValueError(f"Too few numbers in particle row: {ln}")
                x, y, z = map(float, vals[:3])
                xyz.append((x % 10, y % 10, z))
            frames.append(np.asarray(xyz, dtype=float))

    if not frames:
        raise ValueError("No frames parsed. Check the input path/content.")
    return frames, n_expected


def compute_bounds(frames: List[np.ndarray], pad=0.05):
    all_xyz = np.concatenate(frames, axis=0)
    mn = all_xyz.min(axis=0)
    mx = all_xyz.max(axis=0)
    span = np.maximum(mx - mn, 1e-9)
    margin = 0  # span * pad
    return (
        (mn[0] - margin[0], mx[0] + margin[0]),
        (mn[1] - margin[1], mx[1] + margin[1]),
        (mn[2] - margin[2], mx[2] + margin[2]),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Render 2D particle positions (z = color) to image frames."
    )
    ap.add_argument("input", help="Path to the config file.")
    ap.add_argument(
        "-o",
        "--outdir",
        default="frames",
        help="Output directory where frames (PNG) will be saved.",
    )
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    ap.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(512, 512),
        help="Figure size in pixels.",
    )
    ap.add_argument("--marker-size", type=float, default=30.0, help="Marker size.")
    args = ap.parse_args()

    frames, n = parse_config(args.input)
    n_frames = len(frames)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = compute_bounds(frames)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list(
        "green_black", [(0, "#00ff00"), (1, "#000000")]
    )

    w_in, h_in = args.size[0] / args.dpi, args.size[1] / args.dpi
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=args.dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")

    # init scatter
    pts0 = frames[0]
    mask0 = pts0[:, 2] <= 0.5
    scat = ax.scatter(
        pts0[mask0, 0],
        pts0[mask0, 1],
        c=pts0[mask0, 2],
        cmap=cmap,
        s=args.marker_size,
        vmin=0,
        vmax=0.5,
    )

    # No colorbar anymore

    for i, pts in enumerate(frames):
        if i % 10 != 0:
            continue

        mask = pts[:, 2] <= 0.5
        if np.any(mask):
            scat.set_offsets(np.c_[pts[mask, 0], pts[mask, 1]])
            scat.set_array(pts[mask, 2])
        else:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array(np.array([]))

        fname = os.path.join(args.outdir, f"frame_{i:05d}.png")
        fig.savefig(fname, dpi=args.dpi, facecolor=fig.get_facecolor())

    plt.close(fig)
    print(f"Done: saved {n_frames} frames to {args.outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
