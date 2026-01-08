import os
import shutil
import subprocess

def configuration(input_file) :
    for name in os.listdir("."):
        if os.path.isdir(name) and name.startswith("phi="):
            destination = os.path.join(name, input_file)
            shutil.copy(input_file, destination)
            print(f"Copied {input_file} to {destination}")

def runner() :
    for name in os.listdir("."):
        if os.path.isdir(name) and name.startswith("phi="):
            subprocess.run(["python", "multi_bodies.py", "--input-file", "inputfile_blobs.dat"], cwd=name)


def change_clone_file():
    for name in os.listdir("."):
        if os.path.isdir(name) and name.startswith("phi="):
            source = os.path.join(name, "sphere_array.clones")
            destination = os.path.join(name, "initial.sphere_array.clones")
            shutil.copy(source, destination)
            print(f"Copied {source} → {destination}")


import os
import shutil
import subprocess
from typing import List

def _parse_last_block(traj_path: str) -> List[List[float]]:
    """
    Parse the trajectory file and return the last block (list of rows),
    where each row is [x, y, z, qx, qy, qz, qw].
    The file structure is:
        N
        x y z qx qy qz qw
        ... (N rows)
        N
        ...
    """
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    last_block = None
    with open(traj_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    L = len(lines)
    while i < L:
        # Try to read a block length N
        try:
            N = int(float(lines[i]))  # tolerate "6" or "6.0"
        except ValueError:
            i += 1
            continue

        # Ensure there are at least N rows following
        if i + 1 + N <= L:
            block_rows = []
            valid_block = True
            for j in range(i + 1, i + 1 + N):
                parts = lines[j].split()
                if len(parts) < 7:
                    valid_block = False
                    break
                try:
                    row = list(map(float, parts[:7]))  # x y z qx qy qz qw
                except ValueError:
                    valid_block = False
                    break
                block_rows.append(row)

            if valid_block:
                last_block = block_rows
                # advance to after this block
                i = i + 1 + N
                continue

        # if not valid, advance by 1 and keep searching
        i += 1

    if not last_block:
        raise ValueError(f"No valid blocks found in {traj_path}")

    return last_block


def _write_clones_atomic(clones_path: str, rows: List[List[float]]):
    """
    atomic format
    """
    tmp_path = clones_path + ".tmp"
    N = len(rows)
    with open(tmp_path, "w") as f:
        f.write(f"{N}\n")
        for r in rows:
            # ensure exactly 7 numbers per line
            f.write("{:.16g} {:.16g} {:.16g} {:.16g} {:.16g} {:.16g} {:.16g}\n".format(*r))
    os.replace(tmp_path, clones_path)


def update_initial_positions_from_last_step(
    trajectory_filename="run_blobs.sphere_array.config",
    clones_filename="sphere_array.clones",
    keep_backup=True
):
    for name in os.listdir("."):
        if os.path.isdir(name) and name.startswith("phi="):
            traj_path = os.path.join(name, trajectory_filename)
            clones_path = os.path.join(name, clones_filename)
            last_block = _parse_last_block(traj_path)
            if keep_backup:
                backup_path = os.path.join(name, "initial.sphere_array.clones")
                if not os.path.exists(backup_path) and os.path.exists(clones_path):
                    shutil.copy(clones_path, backup_path)
                    print(f"[{name}] Backup saved to {backup_path}")

            _write_clones_atomic(clones_path, last_block)
            print(f"[{name}] Updated {clones_path} with last time-step from {trajectory_filename}")





if __name__ == "__main__":
    configuration("inputfile_blobs.dat")
    configuration("parameters.py")
    configuration("multi_bodies.py")

    runner()
    update_initial_positions_from_last_step(
        trajectory_filename="run_blobs.sphere_array.config",
        clones_filename="sphere_array.clones",
        keep_backup=True 
    )