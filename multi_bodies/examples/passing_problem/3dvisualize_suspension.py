import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Function to read particle positions from a file
def read_positions(file_path):
    positions = []
    with open(file_path, 'r') as file:
        k = 0
        step = 0
        for line in file:
            k += 1
            line = line.split()
            if k == 1:
                Np = int(line[0])
            else:
                if line[0] != str(Np):
                    positions.append(line[0:3])
                else:
                    step += 1
    print(f"Number of particles: {Np}")
    return Np, positions

# Main function
if __name__ == "__main__":
    file_path = 'run_blobs.sphere_array.config'  
    Np, positions = read_positions(file_path)
    Nstep = int(len(positions) // Np)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.cla()
        pos = np.array(positions[i * Np:(i + 1) * Np], dtype=float)
        xs, ys, zs = zip(*pos)

        scatter = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', vmin=0, vmax=25, alpha=0.6)

        ax.set_xlim([-5, 265])
        ax.set_ylim([-5, 25])
        ax.set_zlim([0, 25])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Step {i + 1}/{Nstep}")

    ani = FuncAnimation(fig, update, frames=Nstep, interval=200)

    # Save as a video using ffmpeg
    ani.save('particles_animation.mp4', writer=FFMpegWriter(fps=5))
    print("Video saved as particles_animation.mp4")
