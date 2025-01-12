import sys
import os
import numpy as np
import h5py


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
        vol = mpm_solver.mpm_state.particle_vol.numpy()[0] if hasattr(mpm_solver.mpm_state, 'particle_vol') else 0
        density = mpm_solver.mpm_state.particle_density.numpy()[0] if hasattr(mpm_solver.mpm_state, 'particle_density') else 0

        E = mpm_solver.mpm_model.E.numpy()[0] if hasattr(mpm_solver.mpm_model, 'E') else 0
        nu = mpm_solver.mpm_model.nu.numpy()[0] if hasattr(mpm_solver.mpm_model, 'nu') else 0
        bulk = mpm_solver.mpm_model.bulk.numpy()[0] if hasattr(mpm_solver.mpm_model, 'bulk') else 0

        plastic_viscosity = mpm_solver.mpm_model.plastic_viscosity if hasattr(mpm_solver.mpm_model, 'plastic_viscosity') else 0
        softening = mpm_solver.mpm_model.softening if hasattr(mpm_solver.mpm_model, 'softening') else 0
        yield_stress = mpm_solver.mpm_model.yield_stress.numpy()[0] if hasattr(mpm_solver.mpm_model, 'yield_stress') else 0
        friction_angle = np.tan(np.radians(mpm_solver.mpm_model.friction_angle)) if hasattr(mpm_solver.mpm_model, 'friction_angle') else 0
        alpha = mpm_solver.mpm_model.alpha if hasattr(mpm_solver.mpm_model, 'alpha') else 0

        gravity = mpm_solver.mpm_model.gravitational_accelaration if hasattr(mpm_solver.mpm_model, 'gravitational_accelaration') else [0, 0, 0]
        gx = gravity[0]
        gy = gravity[1]
        gz = gravity[2]

        rpic_damping = mpm_solver.mpm_model.rpic_damping if hasattr(mpm_solver.mpm_model, 'rpic_damping') else 0
        grid_v_damping_scale = mpm_solver.mpm_model.grid_v_damping_scale if hasattr(mpm_solver.mpm_model, 'grid_v_damping_scale') else 0


        table.create_dataset("material_properties",data=np.array([vol, density, E, nu, bulk, plastic_viscosity, softening, yield_stress, friction_angle, alpha, gx, gy, gz, rpic_damping, grid_v_damping_scale]))
