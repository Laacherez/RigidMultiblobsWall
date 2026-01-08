#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
                xyz.append((x, y, z))
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
    return (mn[0]-margin[0], mx[0]+margin[0]), \
           (mn[1]-margin[1], mx[1]+margin[1]), \
           (mn[2]-margin[2], mx[2]+margin[2])

def main():
    ap = argparse.ArgumentParser(description="Render particle positions.")
    ap.add_argument("input", help="Path to the config file.")
    ap.add_argument("-o", "--output", default="trajectories.mp4", help="Output .mp4 path.")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second.")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    ap.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=(1000, 1000), help="Figure size in pixels.")
    ap.add_argument("--marker-size", type=float, default=40.0, help="Marker size.")
    args = ap.parse_args()

    animation.writers['ffmpeg']


    frames, n = parse_config(args.input)
    n_frames = len(frames)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = compute_bounds(frames)

    # Figure
    w_in, h_in = args.size[0] / args.dpi, args.size[1] / args.dpi
    fig = plt.figure(figsize=(w_in, h_in), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=0, azim=45)    


    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))


    title = ax.set_title("Trajectories")
    scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], frames[0][:, 2], s=args.marker_size)

    def init():
        scat._offsets3d = (frames[0][:, 0], frames[0][:, 1], frames[0][:, 2])
        title.set_text(f"frame 1 / {n_frames}")
        return scat, title

    def update(i):
        pts = frames[i]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        title.set_text(f"frame {i+1} / {n_frames}")
        return scat, title

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=1000/args.fps, blit=False)

    FF = animation.writers['ffmpeg']
    writer = FF(fps=args.fps, codec='h264', bitrate=-1)
    anim.save(args.output, writer=writer, dpi=args.dpi)
    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

#python visualizer_from_clones.py "run_blobs.sphere_array.config" -o video.mp4 --fps 30