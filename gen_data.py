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

def sample_points_as_tensor(domain_min = 0.1, domain_max = 0.9, cuboid_size_min = 0.2, cuboid_size_max = 0.4, density = 262144.0, device = "cuda:0"):
    """
    args:
        domain_min: float
            The minimum boundary of the domain in all directions (x, y, z). 
            Particles and cuboids will not extend below this value.

        domain_max: float
            The maximum boundary of the domain in all directions (x, y, z). 
            Particles and cuboids will not extend above this value.

        cuboid_size_min: float
            The minimum size of the cuboid along each axis (x, y, z). 

        cuboid_size_max: float
            The maximum size of the cuboid along each axis (x, y, z). 

        density: float
            The density of particles in the cuboid, specified as points per cubic meter. 
            Determines the number of particles to sample based on the cuboid's volume.

    Returns:
        torch.Tensor:
            A tensor of shape (num_points, 3), where each row represents the (x, y, z) coordinates of a sampled particle
            within the cuboid.
    """
    # cuboidのサイズをランダムに決定
    cuboid_size = np.random.uniform(cuboid_size_min, cuboid_size_max, size=3)
    cuboid_center = np.random.uniform(
        domain_min + cuboid_size / 2, domain_max - cuboid_size / 2, size=3
    )

    # cuboidの範囲を計算
    cuboid_min = cuboid_center - cuboid_size / 2
    cuboid_max = cuboid_center + cuboid_size / 2
    cuboid_volume = np.prod(cuboid_size)

    num_points = int(cuboid_volume * density)

    points = np.random.uniform(cuboid_min, cuboid_max, size=(num_points, 3))

    # torch.tensorに変換
    points_tensor = torch.tensor(points, dtype=torch.float32, device = device)

    return points_tensor


def save_data_at_frame_h5(mpm_solver, dir_name, frame, save_to_ply = True):
    os.umask(0)
    os.makedirs(dir_name, 0o777, exist_ok=True)
    
    fullfilename = dir_name + '/sim_' + str(frame).zfill(10) + '.h5'

    if os.path.exists(fullfilename): os.remove(fullfilename)
    newFile = h5py.File(fullfilename, "w")
    table = newFile.create_group("table")

    x_np = mpm_solver.mpm_state.particle_x.numpy().transpose() # x_np has shape (3, n_particles)

    table.create_dataset("coord_x", data=x_np[0], dtype="float")
    table.create_dataset("coord_y", data=x_np[1], dtype="float")
    table.create_dataset("coord_z", data=x_np[2], dtype="float")

    f_tensor_np = mpm_solver.mpm_state.particle_F_trial.numpy().reshape(-1,9) # shape = (n_particles, 9)
    table.create_dataset("f_tensor", data=f_tensor_np) # deformation grad

    C_np = mpm_solver.mpm_state.particle_C.numpy().reshape(-1,9) # shape = (n_particles, 9)
    table.create_dataset("C", data=C_np) # particle C

    n_particles = len(mpm_solver.mpm_state.particle_vol.numpy())

    material = mpm_solver.mpm_model.material # np.full(length, value)
    table.create_dataset("particle_types",data=np.full(n_particles, material), dtype = int)

    if n_particles>0:
        vol = mpm_solver.mpm_state.particle_vol.numpy()[0]
        density = mpm_solver.mpm_state.particle_density.numpy()[0]
        
        E = mpm_solver.mpm_model.E.numpy()[0]
        nu = mpm_solver.mpm_model.nu.numpy()[0]
        bulk = mpm_solver.mpm_model.bulk.numpy()[0]

        plastic_viscosity = mpm_solver.mpm_model.plastic_viscosity
        softening = mpm_solver.mpm_model.softening
        yield_stress = mpm_solver.mpm_model.yield_stress.numpy()[0]
        friction_angle = np.tan(np.radians(mpm_solver.mpm_model.friction_angle))
        alpha = mpm_solver.mpm_model.alpha

        gravity = mpm_solver.mpm_model.gravitational_accelaration

        gx = gravity[0]
        gy = gravity[1]
        gz = gravity[2]

        rpic_damping = mpm_solver.mpm_model.rpic_damping
        grid_v_damping_scale = mpm_solver.mpm_model.grid_v_damping_scale

        table.create_dataset("material_properties",data=np.array([vol, density, E, nu, bulk, plastic_viscosity, softening, yield_stress, friction_angle, alpha, gx, gy, gz, rpic_damping, grid_v_damping_scale]))

if __name__=="__main__":
    wp.init()
    wp.config.verify_cuda = True

    dvc = "cuda:0"
    mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized

    # n回のシミュレーションを生成
    for i in range(50):
        directory_to_save = f'./sim_results/mpm_{i:04d}'

        position_tensor = sample_points_as_tensor(device = dvc)
        volume_tensor = torch.ones(position_tensor.shape[0])/262144.0

        mpm_solver.load_initial_data_from_torch(tensor_x = position_tensor, tensor_volume = volume_tensor, n_grid = 32, grid_lim = 1.0)

        # Note: You must provide 'density=..' to set particle_mass = density * particle_volume

        material_params = {
            'E': 2000,
            'nu': 0.3,
            "material": "sand",
            'friction_angle': 35, #[15, 17.5, 22.5, 30.0, 37.5, 45.0]
            'g': [0.0, 0.0, -9.8], # [0, 0, -9.8]
            "density": 1800.0
        }
        mpm_solver.set_parameters_dict(material_params)

        mpm_solver.finalize_mu_lam_bulk() # set mu and lambda from the E and nu input

        collider_type = "cut"

        mpm_solver.add_surface_collider((0.0, 0.0, 0.1), ( 0.0, 0.0, 1.0), collider_type, friction = 0.0)
        mpm_solver.add_surface_collider((0.0, 0.0, 0.9), ( 0.0, 0.0,-1.0), collider_type, friction = 0.0)
        mpm_solver.add_surface_collider((0.0, 0.1, 0.0), ( 0.0, 1.0, 0.0), collider_type, friction = 0.0)
        mpm_solver.add_surface_collider((0.0, 0.9, 0.0), ( 0.0,-1.0, 0.0), collider_type, friction = 0.0)
        mpm_solver.add_surface_collider((0.1, 0.0, 0.0), ( 1.0, 0.0, 0.0), collider_type, friction = 0.0)
        mpm_solver.add_surface_collider((0.9, 0.0, 0.0), (-1.0, 0.0, 0.0), collider_type, friction = 0.0)


        save_data_at_frame_h5(mpm_solver, directory_to_save, 0, save_to_ply=True)
        for k in tqdm(range(1, 350)):
            mpm_solver.p2g2p(k, 0.0025, device=dvc)
            save_data_at_frame_h5(mpm_solver, directory_to_save, k, save_to_ply=True)


