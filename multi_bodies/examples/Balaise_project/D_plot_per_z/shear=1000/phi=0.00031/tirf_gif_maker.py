#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
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
                        raise ValueError(f"Inconsistent particle count: {n} (expected {n_expected})")
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
    margin = span * pad
    return (mn[0] - margin[0], mx[0] + margin[0]), \
           (mn[1] - margin[1], mx[1] + margin[1]), \
           (mn[2] - margin[2], mx[2] + margin[2])


def main():
    ap = argparse.ArgumentParser(description="Render 2D particle positions (z = color).")
    ap.add_argument("input", help="Path to the config file.")
    ap.add_argument("-o", "--output", default="trajectories_2d.mp4", help="Output .mp4 path.")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second.")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    ap.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=(1000, 1000), help="Figure size in pixels.")
    ap.add_argument("--marker-size", type=float, default=40.0, help="Marker size.")
    args = ap.parse_args()

    animation.writers['ffmpeg']

    frames, n = parse_config(args.input)
    n_frames = len(frames)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = compute_bounds(frames)

    # --- Custom colormap: bright green (0) → black (1)
    cmap = LinearSegmentedColormap.from_list("green_black", [(0, "#00ff00"), (1, "#000000")])

    # --- Figure setup ---
    w_in, h_in = args.size[0] / args.dpi, args.size[1] / args.dpi
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=args.dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    title = ax.set_title("Trajectories", color="white")

    # First frame
    pts = frames[0]
    mask = pts[:, 2] <= 1.
    scat = ax.scatter(
        pts[mask, 0], pts[mask, 1],
        c=pts[mask, 2],
        cmap=cmap, s=args.marker_size,
        vmin=0, vmax=0.5
    )



    # Colorbar
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label("Z", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

    def init():
        scat.set_offsets(np.c_[frames[0][:, 0], frames[0][:, 1]])
        scat.set_array(frames[0][:, 2])
        return scat, title

    def update(i):
        pts = frames[i]
        mask = pts[:, 2] <= 1.
        if np.any(mask):
            scat.set_offsets(np.c_[pts[mask, 0], pts[mask, 1]])
            scat.set_array(pts[mask, 2])
        else:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array(np.array([]))

        time = (i + 1) * 1e-4
        title.set_text(f"t = {time:.2f} s")
        return scat, title


    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=n_frames, interval=1000 / args.fps, blit=False
    )

    FF = animation.writers['ffmpeg']
    writer = FF(fps=args.fps, codec='h264', bitrate=-1)
    anim.save(args.output, writer=writer, dpi=args.dpi)
    print("Done: saved", args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# python tirf_gif_maker.py run_blobs.sphere_array.config -o tirf.mp4 --fps 300