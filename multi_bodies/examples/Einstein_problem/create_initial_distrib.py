import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_num_particles(Lx, Ly, Lz, a, phi):
    V_box = Lx * Ly * Lz
    V_particle = (4/3) * np.pi * a**3
    return int((phi * V_box) / V_particle)

def is_valid(pos, existing_positions, a, min_dist_factor=2.0):
    """Check if 'pos' is at least 2a away from all existing particles."""
    if len(existing_positions) == 0:
        return True
    dists = np.linalg.norm(existing_positions - pos, axis=1)
    return np.all(dists >= min_dist_factor * a)

def generate_nonoverlapping_positions(N, Lx, Ly, Lz, a, max_attempts=100000):
    positions = []
    attempts = 0
    while len(positions) < N and attempts < max_attempts:
        trial = np.random.uniform(low=[0, 0, 0], high=[Lx, Ly, Lz])
        if is_valid(trial, np.array(positions), a):
            positions.append(trial)
        attempts += 1
    if len(positions) < N:
        raise RuntimeError(f"Only placed {len(positions)} out of {N} particles. Try a lower volume fraction.")
    return np.array(positions)

def save_suspension(positions, output_dir):
    N = positions.shape[0]
    quat = np.tile([[0.0, 1.0, 0.0, 0.0]], (N, 1))
    to_save = np.hstack((positions, quat))

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, 'sphere_array.clones')
    np.savetxt(fname, to_save, header=str(N), comments='')

def visualize_suspension(positions, Lx, Ly, Lz, a):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=50)
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_zlim([0, Lz])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([Lx, Ly, Lz])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sweep over volume fractions and generate suspensions")
    parser.add_argument('--Lx', type=float, default=1.0, help='Box size in X')
    parser.add_argument('--Ly', type=float, default=20.0, help='Box size in Y')
    parser.add_argument('--Lz', type=float, default=1.0, help='Box size in Z')
    parser.add_argument('--a', type=float, default=0.055, help='Particle radius')
    parser.add_argument('--phi_min', type=float, default=3e-5, help='Minimum volume fraction')
    parser.add_argument('--phi_max', type=float, default=6e-3, help='Maximum volume fraction')
    parser.add_argument('--n_phi', type=int, default=5, help='Number of volume fractions to generate')
    parser.add_argument('--out', type=str, default='./', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize each suspension')

    args = parser.parse_args()

    phi_values = np.linspace(args.phi_min, args.phi_max, args.n_phi)

    for phi in phi_values:
        print(f"Generating.")

        N = compute_num_particles(args.Lx, args.Ly, args.Lz, args.a, phi)

        try:
            positions = generate_nonoverlapping_positions(N, args.Lx, args.Ly, args.Lz, args.a)
        except RuntimeError as e:
            print(f"Could not generate for phi = {phi:.4g}: {e}")
            continue

        phi_dirname = f"phi={phi:.4g}"
        full_output_dir = os.path.join(args.out, phi_dirname)
        save_suspension(positions, full_output_dir)
        print(f"Saved to '{full_output_dir}/sphere_array.clones'")

        if args.visualize:
            visualize_suspension(positions, args.Lx, args.Ly, args.Lz, args.a)

