import numpy as np
from pathlib import Path

"""This file shall have all in SI."""

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
    with p.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
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


all_parameters = read_dat("inputfile_blobs.dat")

# Densities (particle/volume), in SI
phi_min = 3e-5                                          # Number
phi_max = 3e-4                                          # Number
how_many_phi = 4                                        # Number
phi_array = np.linspace(phi_min, phi_max, how_many_phi)

# Conversions.
kg_to_mg = 1e6
m_to_um = 1e6
mg_to_kg = 1e-6
um_to_m = 1e-6

# Lengths, in m
box_x_length = all_parameters.get("periodic_length")[0] * um_to_m # m
box_y_width = all_parameters.get("periodic_length")[0] * um_to_m  # m
evanescent_slice_z_height = 500e-9                                # m
particle_radius = all_parameters.get("blob_radius") * um_to_m     # m

# Temperature
kBT = all_parameters.get("kT") * mg_to_kg * um_to_m ** 2
g = all_parameters.get("g") * mg_to_kg * um_to_m # buoyant mass * g
