import numpy as np
import os
import subprocess
import argparse
import csv
import shutil
import concurrent.futures
from functools import partial


def compute_num_particles(Lx, Ly, Lz, a, phi):
    V_box = Lx * Ly * Lz
    V_particle = (4/3) * np.pi * a**3
    return int((phi * V_box) / V_particle)

def is_valid(pos, existing_positions, a, min_dist_factor=2.0):
    if len(existing_positions) == 0:
        return True
    dists = np.linalg.norm(existing_positions - pos, axis=1)
    return np.all(dists >= min_dist_factor * a)

def generate_nonoverlapping_positions(N, Lx, Ly, Lz, a, max_attempts=100000):
    positions = []
    attempts = 0
    while len(positions) < N and attempts < max_attempts:
        trial = np.random.uniform(low=[0, 0, 100], high=[Lx, Ly, 100 + Lz])
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

def run_simulation(multi_bodies_path, input_file, working_dir):
    subprocess.run(
        ['python', multi_bodies_path, '--input-file', input_file],
        cwd=working_dir
    )

def parse_simulation_output(output_file, run_idx, result_writer):
    if not os.path.exists(output_file):
        print(f"Output not found: {output_file}")
        return

    with open(output_file, 'r') as f:
        timestep = -1
        particle_id = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.isdigit():
                timestep += 1
                particle_id = 0
            else:
                parts = list(map(float, line.split()))
                if len(parts) >= 3:
                    x, y, z = parts[:3]
                    result_writer.writerow([run_idx, timestep, particle_id, x, y, z])
                    particle_id += 1

    os.remove(output_file)


def run_single_simulation(run_idx, phi, args, phi_dir, multi_bodies_path, input_file):
    run_dir = os.path.join(phi_dir, f"run_{run_idx:04d}")
    os.makedirs(run_dir, exist_ok=True)
    N = compute_num_particles(args.Lx, args.Ly, args.Lz, args.a, phi)
    try:
        positions = generate_nonoverlapping_positions(N, args.Lx, args.Ly, args.Lz, args.a)
    except RuntimeError as e:
        print(f"Skipping run {run_idx} at phi={phi:.4g}: {e}")
        return []
    save_suspension(positions, run_dir)
    run_simulation(multi_bodies_path, input_file, run_dir)
    output_file = os.path.join(run_dir, 'run_blobs.sphere_array.config')
    output_rows = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            timestep = -1
            particle_id = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    timestep += 1
                    particle_id = 0
                else:
                    parts = list(map(float, line.split()))
                    if len(parts) >= 3:
                        x, y, z = parts[:3]
                        output_rows.append([run_idx, timestep, particle_id, x, y, z])
                        particle_id += 1
        os.remove(output_file)
    else:
        print(f"Output not found: {output_file}")
    return output_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Lx', type=float, default=4.0)
    parser.add_argument('--Ly', type=float, default=4.0)
    parser.add_argument('--Lz', type=float, default=4.0)
    parser.add_argument('--a', type=float, default=0.055)
    parser.add_argument('--phi_min', type=float, default=5e-5)
    parser.add_argument('--phi_max', type=float, default=5e-4)
    parser.add_argument('--n_phi', type=int, default=5)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--out', type=str, default='sweep_runs')
    parser.add_argument('--multi_bodies', type=str, default='multi_bodies.py')
    parser.add_argument('--input_file', type=str, default='inputfile_blobs.dat')

    args = parser.parse_args()

    phi_values = np.linspace(args.phi_min, args.phi_max, args.n_phi)
    multi_bodies_path = os.path.abspath(args.multi_bodies)
    input_file = os.path.abspath(args.input_file)

    for phi in phi_values:
        phi_dirname = f"phi={phi:.4g}"
        phi_dir = os.path.join(args.out, phi_dirname)
        os.makedirs(phi_dir, exist_ok=True)

        result_csv = os.path.join(phi_dir, 'all_runs.csv')
        with open(result_csv, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['run', 'timestep', 'particle_id', 'x', 'y', 'z'])

            with concurrent.futures.ProcessPoolExecutor() as executor:
                run_func = partial(run_single_simulation, phi=phi, args=args,
                                   phi_dir=phi_dir, multi_bodies_path=multi_bodies_path,
                                   input_file=input_file)

                futures = [executor.submit(run_func, run_idx) for run_idx in range(args.num_runs)]
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    for row in results:
                        writer.writerow(row)
