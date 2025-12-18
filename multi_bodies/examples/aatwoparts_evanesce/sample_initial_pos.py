import numpy as np
import os
import parameters

def save_couple_clones(pos2, output_dir, fname="sphere_array.clones"):
    pos2 = np.asarray(pos2, dtype=float).reshape(2, 3)
    quat = np.tile([[0.0, 1.0, 0.0, 0.0]], (2, 1))
    to_save = np.hstack((pos2 * 1e6, quat))  # m to um
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, fname)
    np.savetxt(path, to_save, header="2", comments="")
    return path

def _fmt_um(val_m, ndp=3):
    """Format meters as microns"""
    return f"{val_m*1e6:.{ndp}f}um"

def couple_dirname(fixed_pos, moving_pos, ndp=3):
    x0, y0, z0 = fixed_pos
    x1, y1, z1 = moving_pos
    return (
        f"A({_fmt_um(x0,ndp)},{_fmt_um(y0,ndp)},{_fmt_um(z0,ndp)})"
        f"__B({_fmt_um(x1,ndp)},{_fmt_um(y1,ndp)},{_fmt_um(z1,ndp)})"
    )

def generate_two_particle_couples_deterministic_grid(
    fixed_pos,           # (x0,y0,z0) m
    moving_height,       # z1 m
    x_range,             # (xmin,xmax) m
    y_range,             # (ymin,ymax) m
    nx, ny,              # number of points in x and y
    output_root="./two_particle",
    include_endpoints=True,
    ndp_dirname=3
):


    fixed_pos = np.asarray(fixed_pos, dtype=float).reshape(3,)
    xmin, xmax = x_range
    ymin, ymax = y_range

    xs = np.linspace(xmin, xmax, nx, endpoint=include_endpoints)
    ys = np.linspace(ymin, ymax, ny, endpoint=include_endpoints)

    paths = []
    couples = []

    for x in xs:
        for y in ys:
            moving_pos = np.array([x, y, moving_height], dtype=float)


            subdir = os.path.join(output_root, couple_dirname(fixed_pos, moving_pos, ndp=ndp_dirname))
            path = save_couple_clones(np.vstack([fixed_pos, moving_pos]), subdir)
            paths.append(path)
            couples.append((fixed_pos.copy(), moving_pos.copy()))

    return np.array(couples, dtype=float), paths



import matplotlib.pyplot as plt

def visualize_couples(couples, xdim=None, ydim=None, s=12):
    couples = np.asarray(couples, dtype=float)
    A = couples[:, 0, :]
    B = couples[:, 1, :]

    Ax, Ay, Az = (A[:, 0]*1e6, A[:, 1]*1e6, A[:, 2]*1e6)
    Bx, By, Bz = (B[:, 0]*1e6, B[:, 1]*1e6, B[:, 2]*1e6)


    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.scatter(Bx, By, Bz, s=s, label="Particle B")
    ax3.scatter(Ax[0], Ay[0], Az[0], s=max(3*s, 30), marker="x", label="Particle A")

    ax3.set_xlabel(r"$x\ (\mu\mathrm{m})$")
    ax3.set_ylabel(r"$y\ (\mu\mathrm{m})$")
    ax3.set_zlabel(r"$z\ (\mu\mathrm{m})$")

    if xdim is not None:
        ax3.set_xlim(0, xdim*1e6)
    if ydim is not None:
        ax3.set_ylim(0, ydim*1e6)

    ax3.legend(frameon=False)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    fixed_pos = parameters.particle_a_pos   # Particle A fixed
    moving_height = parameters.particle_b_z               # Particle B fixed z

    x_range = parameters.particle_b_xrange
    y_range = parameters.particle_b_yrange

    couples, files = generate_two_particle_couples_deterministic_grid(
        fixed_pos=fixed_pos,
        moving_height=moving_height,
        x_range=x_range,
        y_range=y_range,
        nx=parameters.how_many_along_x, ny=parameters.how_many_along_y,
        output_root="./",
        ndp_dirname=3
    )

    visualize_couples(couples)
