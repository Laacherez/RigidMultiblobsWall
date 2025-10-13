import numpy as np
import os
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

def compute_boltzmann_length(kT, g) :
    print(kT/(g))
    return kT/(g)

def Peq_nonorm(cell_height, kT, g) : #... ehh
    return np.exp(-( cell_height/compute_boltzmann_length(kT, g)))

def Peq_yesnorm(cell_height, kT, g) : #admittedly, this should fall into the boolistic regime.
    return np.exp(-( cell_height/compute_boltzmann_length(kT, g))) / trapezoid(np.exp(-( cell_height/compute_boltzmann_length(kT, g))), cell_height)

def compute_true_phi(cell_height, evanescence, kT, g, phi) : # Computes the concentration to input before thermalisation. 
    Boltzmann_length = compute_boltzmann_length(kT, g)
    cell_height_array = np.linspace(0, 5* cell_height, 100000)
    Peq = Peq_yesnorm(cell_height_array, kT, g)
    until_lambda = cell_height_array <= evanescence
    N_lambda = trapezoid(Peq[until_lambda], cell_height_array[until_lambda])
    true_phi = phi * evanescence / (N_lambda * 4.7 * Boltzmann_length)
    print(true_phi, N_lambda)
    return true_phi


def compute_num_particles(xdim, ydim, cell_height, particle_radius, phi):
    V_box = xdim * ydim * cell_height
    V_particle = (4/3) * np.pi * particle_radius**3
    return int((phi * V_box) / V_particle)

def is_valid(pos, existing_positions, particle_radius, min_dist_factor=2.0): # With a min_dist_factor of 2.0, particles may be in contact.
    if len(existing_positions) == 0:
        return True
    dists = np.linalg.norm(existing_positions - pos, axis=1)
    return np.all(dists >= min_dist_factor * particle_radius)


def generate_and_save_positions(phi_target, xdim, ydim, evanescence, particle_radius, kT, g, max_attempts=100000, output_dir = "./", show=False): 
    """
    Principal function. 
    Arguments :
    - phi_target : a target concentration for the evanescent slice, 
    - xdim, ydim, evanescence : the three dimenstion of that slice, 
    - particle_radius :the radius of a colloid, 
    - kT : the thermal energy of the medium, 
    - g : the buoyant mass of a colloid,
    - max_attempts : (default: 100000) after how many attempts to stop fitting particles, 
    - output_dir : where to save these positions to their clone files.
    """

    cell_height = 4.7 * compute_boltzmann_length(kT, g) # Compute the height below which 99% of particles lay.
    true_phi = compute_true_phi(cell_height, evanescence, kT, g, phi_target) # Compute the real concentration of that total slice to reach phi_target under evanescence height after thermalistion. (particles sediment)
    N = compute_num_particles(xdim, ydim, cell_height, particle_radius, true_phi) # Compute the corresponding amount of paerticles in the 5lB box.
    positions = []
    attempts = 0

    print(f"Positionning {N} particles... Wait.")
    while len(positions) < N and attempts < max_attempts:
        trial = np.random.uniform(low=[0, 0, 0], high=[xdim, ydim, cell_height])
        if is_valid(trial, np.array(positions), particle_radius):
            positions.append(trial)
        attempts += 1
    if len(positions) < N:
        raise RuntimeError(f"Only placed {len(positions)} out of {N} particles.")
    positions = np.array(positions)
    print(positions)
    save_suspension(positions, output_dir)
    if show == True :
        histogram_of_initial_positions(positions)



def save_suspension(positions, output_dir): # utility formatter and saver for the positions computed in generate_and_save_positions.

    """
    Arguments :
        - output_dir
    Saves to a clone file the original positions for a give parameter set.
    """
    N = positions.shape[0]
    quat = np.tile([[0.0, 1.0, 0.0, 0.0]], (N, 1))
    positions = np.atleast_2d(positions) # In case only 1 particle is present...
    to_save = np.hstack((positions*1e6, quat))

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, 'sphere_array.clones')
    np.savetxt(fname, to_save, header=str(N), comments='')

def histogram_of_initial_positions(positions) :
    xpos = positions[:, 0]
    ypos = positions[:, 1]
    zpos = positions[:, 2]

    plt.hist(xpos, bins=10, label="x", alpha = .5)
    plt.hist(ypos, bins=10, label="y", alpha = .5)
    plt.hist(zpos, bins=10, label="z", alpha = .5)
    plt.ylabel("Starting positions counts")
    plt.xlabel("Positions in um")
    plt.legend(frameon = False)
    plt.show()
    


if __name__ == '__main__':
    import parameters
    kBT = parameters.kBT
    g = parameters.g
    xdim = parameters.box_x_length
    ydim = parameters.box_y_width
    
    
    evanescence = parameters.evanescent_slice_z_height
    particle_radius = parameters.particle_radius

    phi_values = parameters.phi_array #targetted concentration in the evanescent slice

    output_dir = "./"
    for phi in phi_values:
        phi_dirname = f"phi={phi:.4g}"
        phi_dir = os.path.join(output_dir, phi_dirname)
        os.makedirs(phi_dir, exist_ok=True)
        generate_and_save_positions(phi, xdim, ydim, evanescence, particle_radius, kBT, g, max_attempts=100000, output_dir = "./" + phi_dirname, show=True)


#cd multi_bodies/examples/Balaise_project