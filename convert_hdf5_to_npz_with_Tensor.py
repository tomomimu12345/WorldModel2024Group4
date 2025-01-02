import argparse
import pathlib
import glob
import re

import h5py
import numpy as np

#  python3 convert_hdf5_to_npz.py --path sim_results/sand/ sim_results/sand2/ --output sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert hdf5 trajectories to npz.')
    parser.add_argument('--path', nargs="+", help="Path(s) to hdf5 files to consume.")
    parser.add_argument('--ndim', default=3, help="Dimension of input data, default is 2 (i.e., 2D).")
    parser.add_argument('--dt', default=2E-4, help="Time step between position states.")
    parser.add_argument('--output', help="Name of the output file.")
    args = parser.parse_args()

    directories = [pathlib.Path(path) for path in args.path]

    for directory in directories:
        if not directory.exists():
            raise FileExistsError(f"The path {directory} does not exist.")
    print(f"Number of trajectories: {len(directories)}")

    # setup up variables to calculate on-line mean and standard deviation
    # for velocity and acceleration.
    ndim = int(args.ndim)
    if ndim == 2:
        running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
        running_sumsq = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
        running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
    elif ndim == 3:
        running_sum = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0, f_tensor_diff = np.zeros(9))
        running_sumsq = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0, f_tensor_diff = np.zeros(9))
        running_count = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0, acceleration_z=0, f_tensor_diff = 0)
    else:
        raise NotImplementedError        

    trajectories = {}
    for nth_trajectory, directory in enumerate(directories):
        fnames = glob.glob(f"{str(directory)}/*.h5")
        get_fnumber = re.compile(".*\D(\d+).h5")
        fnumber_and_fname = [(int(get_fnumber.findall(fname)[0]), fname) for fname in fnames]
        fnumber_and_fname_sorted = sorted(fnumber_and_fname, key=lambda row: row[0])
        
        # get size of trajectory
        with h5py.File(fnames[0], "r") as f:
            (nparticles,) = f["table"]["coord_x"].shape
        nsteps = len(fnames)

        # allocate memory for trajectory
        # assume number of particles does not change along the rollout.
        positions = np.empty((nsteps, nparticles, ndim), dtype=float)
        print(f"Size of trajectory {nth_trajectory} ({directory}): {positions.shape}")

        # open each file and copy data to positions tensor.
        for nth_step, (_, fname) in enumerate(fnumber_and_fname_sorted):
            with h5py.File(fname, "r") as f:
                for idx, name in zip(range(ndim), ["coord_x", "coord_y", "coord_z"]):
                    positions[nth_step, :, idx] = f["table"][name][:]

        C = np.empty((nsteps, nparticles, ndim * ndim), dtype = float)
        for nth_step, (_, fname) in enumerate(fnumber_and_fname_sorted):
            with h5py.File(fname, "r") as f:
                C[nth_step] = f["table"]["C"]

        f_tensor = np.empty((nsteps, nparticles, ndim * ndim), dtype = float)
        for nth_step, (_, fname) in enumerate(fnumber_and_fname_sorted):
            with h5py.File(fname, "r") as f:
                f_tensor[nth_step] = f["table"]["f_tensor"] 
        
        particle_types = np.empty((nparticles), dtype = int)
        material_properties = np.empty((15),dtype = float)

        _, fname = fnumber_and_fname_sorted[0]  # 最初のファイル
        with h5py.File(fname, "r") as f:
            particle_types[:] = f["table"]["particle_types"] 
            material_properties[:] = f["table"]["material_properties"]

        # compute velocities using finite difference
        # assume velocities before zero are equal to zero
        velocities = np.empty_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1])
        velocities[0] = 0

        f_tensor_diff = np.empty_like(f_tensor)
        f_tensor_diff[1:] = (f_tensor[1:] - f_tensor[:-1])
        f_tensor_diff[0] = 0

        # compute accelerations finite difference
        # assume accelerations before zero are equal to zero
        accelerations = np.empty_like(velocities)
        accelerations[1:] = (velocities[1:] - velocities[:-1])
        accelerations[0] = 0

        # update variables for on-line mean and standard deviation calculation.
        for key in running_sum:
            if key == "velocity_x":
                data = velocities[:,:,0]
            elif key == "velocity_y":
                data = velocities[:,:,1]
            elif key == "velocity_z":
                data = velocities[:,:,2]
            elif key == "acceleration_x":
                data = accelerations[:,:,0]
            elif key == "acceleration_y":
                data = accelerations[:,:,1]
            elif key == "acceleration_z":
                data = accelerations[:,:,2]
            elif key == "f_tensor_diff":
                data = f_tensor_diff # [steps, particle, 9]
            else:
                raise KeyError

            if key == "f_tensor_diff":
                # 要素ごとに合計
                running_sum[key] += np.sum(data, axis=(0,1))  # [9]
                running_sumsq[key] += np.sum(data**2, axis=(0,1))  # [9]
                running_count[key] += data.shape[0] * data.shape[1] # steps * particles
            else:
                # スカラー値の合計
                running_sum[key] += np.sum(data)
                running_sumsq[key] += np.sum(data**2)
                running_count[key] += np.size(data)

        trajectories[str(directory)] = {
            "positions":positions, 
            "particle_types": particle_types,
            "material_properties": material_properties,
            "C": C,
            "f_tensor": f_tensor
        }

    # compute online mean and standard deviation.
    print("Statistis across all trajectories:")
    for key in running_sum:
        mean = running_sum[key] / running_count[key]
        std = np.sqrt((running_sumsq[key] - running_sum[key]**2/running_count[key]) / (running_count[key] - 1))
        if isinstance(mean, np.ndarray):
            # 配列の場合
            mean_str = np.array2string(mean, formatter={'float_kind': lambda x: f"{x:.4E}"})
            std_str = np.array2string(std, formatter={'float_kind': lambda x: f"{x:.4E}"})
            print(f"  {key}: mean={mean_str}, std={std_str}")
        else:
            # スカラーの場合
            print(f"  {key}: mean={mean:.4E}, std={std:.4E}")

    # for key in trajectories:
    #     for i,array in enumerate(trajectories[key]):
    #         print(f"{i}:{array}")
    np.savez_compressed(args.output, **trajectories)
    print(f"Output written to: {args.output}")
