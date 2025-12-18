import numpy as np
from pathlib import Path

"""This file shall have all in SI how shall not be."""

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


all_parameters = read_dat("inputfile_shear_blobs.dat")
num_runs = 10

# Conversions.
kg_to_mg = 1e6
m_to_um = 1e6
mg_to_kg = 1e-6
um_to_m = 1e-6

# Lengths, in m
box_x_length = all_parameters.get("periodic_length")[0] * um_to_m # m
box_y_width = all_parameters.get("periodic_length")[1] * um_to_m  # m
evanescent_slice_z_height = 500e-9                                # m
particle_radius = all_parameters.get("blob_radius") * um_to_m     # m

# Temperature
kBT = all_parameters.get("kT") * mg_to_kg * um_to_m ** 2
g = all_parameters.get("g") * mg_to_kg * um_to_m # buoyant mass * g

# Times
time_step = all_parameters.get("dt")

# Shears
shear_rate = all_parameters.get("shear")[2]
# Debye
B = 3.
lD = all_parameters.get("debye_length_wall") * um_to_m # m

# Initial particle positions], maybe remove x range lol. i was real tired on monday
particle_a_pos = (2.5e-6, 2.5e-6, 0.40e-6)
particle_b_xrange = (2.0e-6, 2.5e-6)
particle_b_yrange = (2.0e-6, 2.5e-6)
particle_b_z = 0.40e-6   - 2 * particle_radius

how_many_along_x = 3
how_many_along_y = 3