import os
import subprocess
import argparse
import csv
import shutil

def run_all_clones(clones_root, multi_bodies_path, input_file_name, num_runs):
    multi_bodies_path = os.path.abspath(multi_bodies_path)
    input_file = os.path.abspath(input_file_name)

    subdirs = [d for d in os.listdir(clones_root) if os.path.isdir(os.path.join(clones_root, d))]

    for subdir in sorted(subdirs):
        full_path = os.path.join(clones_root, subdir)

        if not os.path.exists(input_file):
            print(f"Input file does not exist: {input_file}")
            break


        csv_path = os.path.join(full_path, 'all_runs.csv')
        with open(csv_path, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['run', 'timestep', 'particle_id', 'x', 'y', 'z'])

            for run_idx in range(num_runs):
                print(f"Running multi_bodies.py in {subdir}, run {run_idx + 1}/{num_runs}")

                # Run the script
                subprocess.run(
                    ['python', multi_bodies_path, '--input-file', input_file],
                    cwd=full_path
                )

                # Parse and save output
                output_file = os.path.join(full_path, 'run_blobs.sphere_array.config')
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f_in:
                        timestep = -1
                        particle_id = 0
                        for line in f_in:
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
                                    writer.writerow([run_idx, timestep, particle_id, x, y, z])
                                    particle_id += 1
                                else:
                                    print(f"Warning: malformed line in {output_file}: {line}")
                    os.remove(output_file)  
                else:
                    print(f"Expected output {output_file} not found after run {run_idx} in {subdir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run multi-bodies.py multiple times in all clone subfolders")
    parser.add_argument('--clones_root', default='pairs_output', help='Directory containing the clone folders')
    parser.add_argument('--multi_bodies', default='multi_bodies.py', help='Path to multi-bodies.py')
    parser.add_argument('--input_file', default='inputfile_blobs.dat', help='Path to the input file')
    parser.add_argument('--num_runs', type=int, default=10000, help='Number of times to run multi-bodies.py per clone')

    args = parser.parse_args()
    run_all_clones(args.clones_root, args.multi_bodies, args.input_file, args.num_runs)
