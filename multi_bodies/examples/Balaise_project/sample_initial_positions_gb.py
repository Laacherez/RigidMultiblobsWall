import numpy as np
import os
from scipy.integrate import trapezoid, cumulative_trapezoid
import matplotlib.pyplot as plt
import shutil
import seaborn as sns

custom_params = {
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.markeredgecolor": "k",
    "lines.markeredgewidth": 1.25,
    "figure.dpi": 200,
    "text.usetex": True,
    "font.family": "serif",
}
sns.set_theme(context="talk", style="ticks", rc=custom_params)


def compute_boltzmann_length(kT, g):
    return kT / g


def _build_inverse_cdf(H, R, kT, g, B, debye_length, ngrid=20000):
    ell = compute_boltzmann_length(kT, g)
    zmin, zmax = R, H - R
    if not (zmax > zmin):
        raise ValueError(f"Invalid vertical domain: R={R}, H={H}.")

    z = np.linspace(zmin, zmax, ngrid)
    # Dimensionless potential U/kT
    U_over_kT = (z / ell) + B * np.exp(-z / debye_length)
    pdf = np.exp(-U_over_kT)

    # Normalize via CDF
    cdf = cumulative_trapezoid(pdf, z, initial=0.0)
    Z = cdf[-1]
    cdf /= Z

    def sample_z(rng=np.random):
        u = rng.random()
        return np.interp(u, cdf, z)

    return sample_z


def Peq_nonorm(cell_height, kT, g, B, debye_length):
    ell = compute_boltzmann_length(kT, g)
    return np.exp(-(B * np.exp(-cell_height / debye_length) + cell_height / ell))


def Peq_yesnorm(cell_height, kT, g, B, debye_length):
    Peq = Peq_nonorm(cell_height, kT, g, B, debye_length)
    return Peq / trapezoid(Peq, cell_height)


def compute_true_phi(cell_height, evanescence, kT, g, B, debye_length, phi):
    z = np.linspace(0, cell_height, 100000)
    Peq = Peq_yesnorm(z, kT, g, B, debye_length)
    N_lambda = trapezoid(Peq[z <= evanescence], z[z <= evanescence])
    true_phi = phi * evanescence / (N_lambda * cell_height)
    return true_phi


def compute_num_particles(xdim, ydim, cell_height, particle_radius, phi):
    V_box = xdim * ydim * cell_height
    V_particle = (4 / 3) * np.pi * particle_radius**3
    return int((phi * V_box) / V_particle)


def is_valid(pos, existing_positions, particle_radius, min_dist_factor=2.0):
    if len(existing_positions) == 0:
        return True
    d = existing_positions - pos
    dists = np.linalg.norm(d, axis=1)
    return np.all(dists >= min_dist_factor * particle_radius)


def visualize_positions_3d(positions, xdim, ydim, H):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        positions[:, 0] * 1e6, positions[:, 1] * 1e6, positions[:, 2] * 1e6, s=30
    )
    ax.set_xlim(0, xdim * 1e6)
    ax.set_ylim(0, ydim * 1e6)
    ax.set_zlim(0, H * 1e6)
    ax.set_xlabel("$x (\mu \mathrm{m})$")
    ax.set_ylabel("$y (\mu \mathrm{m})$")
    ax.set_zlabel("$z (\mu \mathrm{m})$")
    ax.view_init(elev=0, azim=270)

    plt.tight_layout()
    plt.show()


def generate_and_save_positions(
    phi_target,
    xdim,
    ydim,
    evanescence,
    particle_radius,
    kT,
    g,
    B,
    debye_length,
    max_attempts=100000,
    output_dir="./",
    show=False,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng

    lB = compute_boltzmann_length(kT, g)
    H = 5.0 * lB

    true_phi = compute_true_phi(H, evanescence, kT, g, B, debye_length, phi_target)
    N = compute_num_particles(xdim, ydim, H, particle_radius, true_phi)

    sample_z = _build_inverse_cdf(H, particle_radius, kT, g, B, debye_length)

    positions = []
    attempts = 0

    print(f"Placing {N} particles with rejection.")
    while len(positions) < N and attempts < max_attempts:
        trial = np.array(
            [rng.uniform(0.0, xdim), rng.uniform(0.0, ydim), sample_z(rng)]
        )
        if is_valid(trial, np.array(positions), particle_radius):
            positions.append(trial)
        attempts += 1

    if len(positions) < N:
        raise RuntimeError(
            f"Only placed {len(positions)} out of {N} (attempts={attempts})."
        )

    positions = np.array(positions)
    save_suspension(positions, output_dir)
    if show:
        histogram_of_initial_positions(positions, B, debye_length, lB)
        visualize_positions_3d(positions, xdim, ydim, H)


def save_suspension(positions, output_dir):
    N = positions.shape[0]
    quat = np.tile([[0.0, 1.0, 0.0, 0.0]], (N, 1))
    positions = np.atleast_2d(positions)
    to_save = np.hstack((positions * 1e6, quat))
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "sphere_array.clones")
    np.savetxt(fname, to_save, header=str(N), comments="")


def histogram_of_initial_positions(positions, B, debye_length, lB):
    xpos = positions[:, 0]
    ypos = positions[:, 1]
    zpos = positions[:, 2]

    bins = 20
    plt.hist(xpos, bins=bins, label="x", alpha=0.5, density=True)
    plt.hist(ypos, bins=bins, label="y", alpha=0.5, density=True)
    plt.hist(zpos, bins=bins, label="z", alpha=0.5, density=True)
    plt.ylabel("P(q)")
    plt.xlabel("Position")

    zarray = np.linspace(0, 5 * lB, 1000)
    P_eq_nonorm = np.exp(-(B * np.exp(-zarray / debye_length) + zarray / lB))
    P_eq_norm = P_eq_nonorm / trapezoid(P_eq_nonorm, zarray)
    # plt.xlim(0, 1e-6)
    plt.plot(zarray, P_eq_norm)
    plt.legend(frameon=False)
    plt.show()


def configuration(input_file):
    for name in os.listdir("."):
        if os.path.isdir(name) and name.startswith("phi="):
            destination = os.path.join(name, input_file)
            shutil.copy(input_file, destination)
            print(f"Copied {input_file} to {destination}")


if __name__ == "__main__":
    import parameters

    kBT = parameters.kBT
    g = parameters.g
    xdim = parameters.box_x_length
    ydim = parameters.box_y_width
    B = parameters.B
    debye_length = parameters.lD

    evanescence = parameters.evanescent_slice_z_height
    particle_radius = parameters.particle_radius

    phi_values = parameters.phi_array  # targetted concentration in the evanescent slice

    output_dir = "./"

    for phi in phi_values:
        phi_dirname = f"phi={phi:.4g}"
        phi_dir = os.path.join(output_dir, phi_dirname)
        os.makedirs(phi_dir, exist_ok=True)
        generate_and_save_positions(
            phi,
            xdim,
            ydim,
            evanescence,
            particle_radius,
            kBT,
            g,
            B,
            debye_length,
            max_attempts=100000,
            output_dir="./" + phi_dirname,
            show=True,
        )

    configuration("parameters.py")
    configuration("multi_bodies.py")
# cd multi_bodies/examples/Balaise_project
