#!/usr/bin/env python3
import numpy as np
import os
from pathlib import Path


def _coerce(token: str):
    try:
        f = float(token)
        i = int(f)
        return i if f == i else f
    except ValueError:
        return token


def read_dat(path):
    data = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            key, values = parts[0], parts[1:]
            if not values:
                data[key] = True
            elif len(values) == 1:
                data[key] = _coerce(values[0])
            else:
                data[key] = [_coerce(v) for v in values]
    return data


def save_suspension(positions_m, output_dir):
    """
    positions_m: (N,3) in meters
    Saves 'sphere_array.clones' with positions in microns + quaternion, header=N.
    """
    positions_m = np.atleast_2d(positions_m)
    N = positions_m.shape[0]
    quat = np.tile([[0.0, 1.0, 0.0, 0.0]], (N, 1))
    to_save = np.hstack((positions_m * 1e6, quat))  # m -> µm

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "sphere_array.clones")
    np.savetxt(fname, to_save, header=str(N), comments="")


def generate_pair_positions(xdim, ydim, particle_radius, h, d):

    R = particle_radius

    # z0 = R + h
    z0 = h

    x_center = 0.5 * xdim
    y0 = 0.5 * ydim

    x1 = x_center - 0.5 * d
    x2 = x_center + 0.5 * d

    if x1 < R or x2 > xdim - R:
        raise ValueError(
            f"d={d} too large for xdim={xdim} with margin R={R}. "
            f"Got x1={x1}, x2={x2}."
        )

    return np.array([[x1, y0, z0], [x2, y0, z0]], dtype=float)


def generate_pairs_over_grid(xdim, ydim, particle_radius, harray, darray, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    n_ok = 0
    n_fail = 0

    for h in harray:
        for d in darray:
            h_um = h * 1e6
            d_um = d * 1e6
            case_dir = os.path.join(base_dir, f"h={h_um:.6g}um_d={d_um:.6g}um")
            try:
                positions = generate_pair_positions(xdim, ydim, particle_radius, h, d)
                save_suspension(positions, case_dir)
                n_ok += 1
            except ValueError as e:
                # Skip invalid combos cleanly
                n_fail += 1
                # If you want to see failures, uncomment:
                print(f"Skipping h={h_um:.4g} um, d={d_um:.4g} um: {e}")

    print(f"Done. Wrote {n_ok} cases to '{base_dir}'. Skipped {n_fail} invalid cases.")


if __name__ == "__main__":
    dat_path = "inputfile.dat"
    all_parameters = read_dat(dat_path)

    # Conversions
    mg_to_kg = 1e-6
    um_to_m = 1e-6

    # Lengths (in m)
    periodic_length = all_parameters.get("periodic_length")
    if periodic_length is None or len(periodic_length) < 2:
        raise KeyError(
            "Expected 'periodic_length' with at least two values in the .dat file."
        )
    box_x_length = periodic_length[0] * um_to_m
    box_y_width = periodic_length[1] * um_to_m

    particle_radius = all_parameters.get("blob_radius") * um_to_m

    # Build log-spaced arrays
    dmin = 1.1 * particle_radius
    dmax = 10.0 * particle_radius
    Nd = 2
    darray = np.logspace(np.log10(dmin), np.log10(dmax), Nd)

    hmin = dmin
    hmax = dmax
    Nh = Nd
    harray = np.logspace(np.log10(hmin), np.log10(hmax), Nh)

    # Output base directory
    base_dir = "./pairs"

    generate_pairs_over_grid(
        xdim=box_x_length,
        ydim=box_y_width,
        particle_radius=particle_radius,
        harray=harray,
        darray=darray,
        base_dir=base_dir,
    )
