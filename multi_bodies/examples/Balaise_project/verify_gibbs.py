import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from typing import List, Tuple

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


def parse_config(
    path: str,
) -> Tuple[List[np.ndarray], int]:  # shamelessly using the visualizer code lol
    positions: List[np.ndarray] = []
    n_expected = None

    with open(path, "r", encoding="utf-8") as f:
        lines = (ln.strip() for ln in f)
        while True:
            for ln in lines:
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) == 1 and parts[0].lstrip("-").isdigit():
                    n = int(parts[0])
                    if n <= 0:
                        raise ValueError(f"Invalid particle count: {n}")
                    if n_expected is None:
                        n_expected = n
                    elif n_expected != n:
                        raise ValueError(
                            f"Inconsistent particle count: {n} (expected {n_expected})"
                        )
                    break
                else:
                    raise ValueError(f"Expected particle count line, got: {ln}")
            else:
                break

            xyz = []
            for _ in range(n):
                ln = next(lines).strip()
                vals = ln.split()
                if len(vals) < 3:
                    raise ValueError(f"Too few numbers in particle row: {ln}")
                x, y, z = map(float, vals[:3])
                # print(x)

                xyz.append((x, y, z))
                # print(x, y, z)
            positions.append(np.asarray(xyz, dtype=float))
        # print(positions)

    return positions, n_expected


def verify_gibbs_boltzmann(positions, timestep_index=5000):
    positions_t = positions[timestep_index]
    xpos = positions_t[:, 0]
    ypos = positions_t[:, 1]
    zpos = positions_t[:, 2]

    zpos_pos = zpos[zpos > 0]

    z_min = 0.1
    z_max = zpos_pos.max()

    nbins = 20
    bins = np.geomspace(z_min, z_max, nbins + 1)

    zarray = np.geomspace(z_min, 10 * 2.20, 1000)
    P_eq_nonorm = np.exp(-(3 * np.exp(-zarray / 0.01) + zarray / 2.2))
    P_eq_norm = P_eq_nonorm / trapezoid(P_eq_nonorm, zarray)

    plt.figure()
    plt.plot(zarray, P_eq_norm, label="theory")

    plt.hist(
        zpos_pos,
        bins=bins,
        label="z (log bins)",
        alpha=0.5,
        density=True,
    )

    plt.xscale("log")
    # plt.yscale("log")

    plt.xlabel(r"$z\ (\mu \mathrm{m})$")
    plt.ylabel(r"$P(z)$")
    plt.legend(frameon=False)

    plt.savefig(f"histogram_z_{timestep_index}.pdf", transparent=True)
    plt.show()


if __name__ == "__main__":
    positions, n = parse_config(path="./run_blobs.sphere_array.config")
    verify_gibbs_boltzmann(positions, timestep_index=10000)
