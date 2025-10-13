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




if __name__ == "__main__":
    configuration("inputfile_blobs.dat")
    configuration("parameters.py")
    configuration("multi_bodies.py")

    runner()