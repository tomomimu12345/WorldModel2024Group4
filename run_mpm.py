import sys
import os

warp_mpm_dir = "./warp-mpm/"

if warp_mpm_dir not in sys.path:
    sys.path.append(warp_mpm_dir)


from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
import warp as wp
from tqdm import tqdm
import numpy as np
from utils.output_h5 import save_data_at_frame_h5

if __name__=="__main__":
    wp.init()
    wp.config.verify_cuda = True

    dvc = "cuda:0"
    mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized
    
    # You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
    mpm_solver.load_from_sampling("./warp-mpm/sand_column.h5", n_grid = 150, device=dvc)  # n_grid=32, grid_lim = 1.0

    # Or load from torch tensor (also position and volume)
    # Here we borrow the data from h5, but you can use your own
    volume_tensor = torch.ones(mpm_solver.n_particles) * 2.5e-8 # /262144.0
    position_tensor = mpm_solver.export_particle_x_to_torch()

    mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor) # n_grid = 32, grid_lim = 1.0

    # Note: You must provide 'density=..' to set particle_mass = density * particle_volume

    material_params = {
        'E': 2000,
        'nu': 0.2, # 0.3
        "material": "sand",
        'friction_angle': 35, #[15, 17.5, 22.5, 30.0, 37.5, 45.0]
        'g': [0.0, 0.0, -4.0], # [0, 0, -9.8]
        "density": 200.0 # 1800.0
    }
    mpm_solver.set_parameters_dict(material_params)

    mpm_solver.finalize_mu_lam_bulk() # set mu and lambda from the E and nu input

    mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', friction = 0.0) # 'slip' 


    directory_to_save = './sim_results/sand'

    save_data_at_frame_h5(mpm_solver, directory_to_save, 0, save_to_ply=True)
    for k in tqdm(range(1,10)):
        mpm_solver.p2g2p(k, 0.002, device=dvc)
        save_data_at_frame_h5(mpm_solver, directory_to_save, k, save_to_ply=True)


