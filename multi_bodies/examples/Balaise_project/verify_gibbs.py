import numpy as np
import os
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import argparse
import os
import sys
from typing import List, Tuple


def parse_config(path: str) -> Tuple[List[np.ndarray], int]: #shamelessly using the visualizer code lol
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
                        raise ValueError(f"Inconsistent particle count: {n} (expected {n_expected})")
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


def verify_gibbs_boltzmann(positions, timestep_index = 5000):
    # positions = np.array(positions)

    positions = positions[timestep_index]
    xpos = positions[:, 0]
    ypos = positions[:, 1]
    zpos = positions[:, 2]


    # plt.hist(xpos, bins=10, label="x", alpha = .5)
    # plt.hist(ypos, bins=10, label="y", alpha = .5)
    zarray = np.linspace(0, 5 * 2.20, 1000)
    P_eq_nonorm = np.exp(-(5 * np.exp(-zarray/0.01) + zarray/2.2))

    P_eq_norm = P_eq_nonorm / trapezoid(P_eq_nonorm, zarray)
    plt.plot(zarray, P_eq_norm)
    plt.hist(zpos, bins=20, label="z", alpha = .5, density = True)
    plt.ylabel("Counts")
    plt.xlabel("Positions in um")
    plt.yscale('log')
    plt.legend(frameon = False)
    plt.show()



if __name__ == '__main__':
    positions, n = parse_config(path = "./end.config")
    verify_gibbs_boltzmann(positions, timestep_index=0)