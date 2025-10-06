import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
import numpy as np

def read_positions_from_csv(file_path, run_id=0):
    df = pd.read_csv(file_path)
    df = df[df['run'] == run_id].sort_values(by=['timestep', 'particle_id'])

    timesteps = df['timestep'].unique()
    particle_ids = df['particle_id'].unique()
    positions_per_frame = []

    for t in timesteps:
        frame = df[df['timestep'] == t][['x', 'y', 'z']].values
        positions_per_frame.append(frame)

    return positions_per_frame, len(particle_ids), len(timesteps)

if __name__ == "__main__":
    file_path = '/Users/juls/RigidMultiblobsWall-1/multi_bodies/examples/Einstein_problem/sweep_runs/phi=0.0005/all_runs.csv'  # CSzV with columns: run, timestep, particle_id, x, y, z
    run_id = 0  

    positions, Np, Nstep = read_positions_from_csv(file_path, run_id=run_id)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    def update(i):
        ax1.cla()
        ax2.cla()

        pos = positions[i]
        xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
 
        ax1.scatter(xs, zs, c=zs, vmin=0, vmax=25, cmap='viridis', alpha=0.5)
        ax2.scatter(xs, ys, c=zs, vmin=0, vmax=25, cmap='viridis', alpha=0.5)

        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([0, 25])

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_xlim([-10, 10])
        ax2.set_ylim([-5, 25])
        fig.suptitle(f"Timestep {i}")

    ani = FuncAnimation(fig, update, frames=Nstep, interval=200)

    ani.save('particles_animation.gif', writer=PillowWriter(fps=5))
    print("GIF saved as particles_animation.gif")
