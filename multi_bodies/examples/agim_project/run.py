import shutil
import subprocess
from pathlib import Path


def iter_case_dirs(base_dir="pairs"):
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory '{base_dir}' not found.")
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("h=") and "_d=" in p.name:
            yield p


def configuration(files_to_copy, base_dir="pairs"):
    files_to_copy = [Path(f) for f in files_to_copy]
    for f in files_to_copy:
        if not f.exists():
            raise FileNotFoundError(f"File to copy not found: {f}")

    for case_dir in iter_case_dirs(base_dir):
        for f in files_to_copy:
            shutil.copy(f, case_dir / f.name)
        print(f"Copied {len(files_to_copy)} files to {case_dir}")


def runner(
    base_dir="pairs",
    python_exe="python",
    n_traj=10,
    output_name="run_blobs.sphere_array.config",
    out_subdir="trajectories",
):
    """
    For each case directory:
      - run multi_bodies.py n_traj times
      - rename/move the produced output file to a unique name
    """
    for case_dir in iter_case_dirs(base_dir):
        traj_dir = case_dir / out_subdir
        traj_dir.mkdir(exist_ok=True)

        for i in range(n_traj):
            cmd = [python_exe, "multi_bodies.py", "--input-file", "inputfile.dat"]
            subprocess.run(cmd, cwd=str(case_dir), check=True)

            produced = case_dir / output_name
            if not produced.exists():
                raise FileNotFoundError(
                    f"Expected output '{output_name}' not found after run "
                    f"{i+1}/{n_traj} in {case_dir}"
                )

            renamed = traj_dir / f"traj_{i:04d}.config"

            # If you rerun the script, don't silently overwrite previous results
            if renamed.exists():
                raise FileExistsError(
                    f"{renamed} already exists. Delete it or change naming."
                )

            produced.replace(renamed)  # atomic move on same filesystem

        print(f"{case_dir}: wrote {n_traj} trajectories to {traj_dir}")


if __name__ == "__main__":
    configuration(
        files_to_copy=[
            "inputfile.dat",
            "multi_bodies.py",
        ],
        base_dir="pairs",
    )

    runner(base_dir="pairs", n_traj=20)
