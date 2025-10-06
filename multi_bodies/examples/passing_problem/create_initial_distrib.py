import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_coords(n, L, start=0.0, mode="start"):
    """
    Generate coordinates along an axis.
    If n==1: returns [start] or [start + L/2] depending on mode.
    If n>1: returns linspace from start to start + L.
    """
    if n == 1:
        if mode == "start":
            return np.array([start])
        elif mode == "center":
            return np.array([start + L / 2])
        else:
            raise ValueError("Invalid mode for make_coords: choose 'start' or 'center'.")
    else:
        return np.linspace(start, start + L, n)

def save_pair(fixed_pos, cuboid_pos, output_dir):
    Np = 2
    quat = np.array([[0.0, 1.0, 0.0, 0.0]])
    pos = np.vstack((fixed_pos, cuboid_pos))
    quat_all = np.vstack((quat, quat))
    to_save = np.hstack((pos, quat_all))

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, 'sphere_array.clones')
    np.savetxt(fname, to_save, header=str(Np), comments='')

def generate_pairs(Nx, Lx, Ny, Ly, Nz, Lz, a, base_output):
    """
    Generate one fixed pos + one passing particle per folder.
    Cuboid spans from -Lx to 0 along x, 0 to Ly along y, 0 to Lz along z.
    """
    fixed_pos = np.array([[0.0, 0.0, 100.0]])
    count = 0

    x_vals = make_coords(Nx, Lx, start=-Lx, mode="start")
    y_vals = make_coords(Ny, Ly, start=0.0, mode="start")
    z_vals = make_coords(Nz, Lz, start=100.0 + a, mode="start")

    for i, xi in enumerate(x_vals):
        for j, yj in enumerate(y_vals):
            for k, zk in enumerate(z_vals):
                cuboid_pos = np.array([[xi, yj, zk]])
                folder_name = os.path.join(base_output, f"pair_{count:04d}_i{i}_j{j}_k{k}")
                save_pair(fixed_pos, cuboid_pos, folder_name)
                count += 1

    print(f"Generated {count} pair configurations...")

def visualize_positions(Nx, Lx, Ny, Ly, Nz, Lz, a):
    """
    Visualize fixed + all cuboid positions with labels
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    fixed_pos = np.array([[0.0, 0.0, 100.0]])
    ax.scatter(fixed_pos[:, 0], fixed_pos[:, 1], fixed_pos[:, 2], c='red', s=150, label='fixed particle')
    ax.text(0.0, 0.0, 100.0 + 0.5*a, "fixed", color='red', fontsize=10, ha='center')

    x_vals = make_coords(Nx, Lx, start=-Lx, mode="start")
    y_vals = make_coords(Ny, Ly, start=0.0, mode="start")
    z_vals = make_coords(Nz, Lz, start=100.0 + a, mode="start")

    count = 0
    for i, xi in enumerate(x_vals):
        for j, yj in enumerate(y_vals):
            for k, zk in enumerate(z_vals):
                ax.scatter([xi], [yj], [zk], c='blue', s=80)
                ax.text(xi+ 0.2*a, yj, zk, f"{count:04d}", fontsize=8, ha='center')
                count += 1

    ax.set_xlabel('X (flow direction)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Initial Configuration (Fixed + Cuboid Particles)')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate pairwise clones for cuboid behind a fixed sphere")
    parser.add_argument('--Nx', type=int, default=1, help='Particles along -x')
    parser.add_argument('--Lx', type=float, default=2.2, help='Length of cuboid along -x')
    parser.add_argument('--Ny', type=int, default=1, help='Particles along +y')
    parser.add_argument('--Ly', type=float, default=0.0, help='Length along +y')
    parser.add_argument('--Nz', type=int, default=20, help='Particles along +z')
    parser.add_argument('--Lz', type=float, default=3.3, help='Length along +z')
    parser.add_argument('--a', type=float, default=1.0, help='Particle radius')
    parser.add_argument('--out', type=str, default='pairs_output', help='Output directory')

    args = parser.parse_args()

    visualize_positions(args.Nx, args.Lx, args.Ny, args.Ly, args.Nz, args.Lz, args.a)
    generate_pairs(args.Nx, args.Lx, args.Ny, args.Ly, args.Nz, args.Lz, args.a, args.out)

    print(f"Nx={args.Nx}, Lx={args.Lx}, a={args.a}")
