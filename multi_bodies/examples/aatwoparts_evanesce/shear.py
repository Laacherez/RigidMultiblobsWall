import os
import shutil
import subprocess
from pathlib import Path
import parameters

def configuration(input_file) :
    for name in os.listdir("."):
        if os.path.isdir(name) and name.startswith("A("):
            destination = os.path.join(name, input_file)
            shutil.copy(input_file, destination)
            print(f"Copied {input_file} to {destination}")

# def runner() :
#     for name in os.listdir("."):
#         if os.path.isdir(name) and name.startswith("A("):
#             subprocess.run(["python", "multi_bodies.py", "--input-file", "inputfile_shear_blobs.dat"], cwd=name)

def runner(cwd) :
    subprocess.run(["python", "multi_bodies.py", "--input-file", "inputfile_shear_blobs.dat"], cwd=cwd)


# def run_many(n=2):
#     for i in range(1, n + 1):
#         runner()

#         src = OUT_DIR / OUT_NAME
#         if not src.exists():
#             raise FileNotFoundError(f"Expected output not found: {src}")

#         dst = OUT_DIR / f"run_blobs.sphere_array_{i:02d}.config"
#         src.rename(dst)




def run_all(num_runs, BASE):
    for subdir in sorted(
        p for p in BASE.iterdir()
        if p.is_dir() and p.name.startswith("A(")
    ):
        
        
        
        
        for i in range(1, num_runs + 1):
            runner(subdir)
            os.chdir(subdir)
            src = Path("run_blobs.sphere_array.config")

            dst = subdir / f"run_blobs.sphere_array_{i:02d}.config"
            src.rename(dst)
            os.chdir(BASE)

        



if __name__ == "__main__":
    BASE = Path("/Users/juls/RigidMultiblobsWall-1/multi_bodies/examples/aatwoparts_evanesce")
    configuration("inputfile_shear_blobs.dat")
    configuration("multi_bodies.py")

    run_all(parameters.num_runs, BASE)