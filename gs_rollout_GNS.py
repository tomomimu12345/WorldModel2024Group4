import collections
import json
import os
import pickle
import glob
import re
import sys


import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import matplotlib.pyplot as plt

from absl import flags
from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('./efficient-kan/src')

from gns_with_tensor import learned_simulator
from gns_with_tensor import noise_utils
from gns_with_tensor import reading_utils
from gns_with_tensor import data_loader
from gns_with_tensor import distribute
from utils.tensor_utils import *

import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
INPUT_TENSOR_SEQUENCE_LENGTH = 1 # last 1 tensor difference f_tensor
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 7

loss_weight = {
  "acc": 1.,
  "f_tensor_diff": 1.,
  "C_diff": 1.,
}

use_material_properties_list = np.array([False]*15)
# use_material_properties_list[8] = True

use_f_tensor = True
use_C_tensor = False

use_KAN = False
mlp_hidden_dim=128

@timer
def rollout(
        FLAGS: dict,
        simulator: learned_simulator.LearnedSimulator,
        covariance: torch.tensor,
        position: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        device: torch.device,
        f_tensor: torch.tensor = None,
        C: torch.tensor = None): # f_tensorとCは使わない場合を考慮する
  """
  Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    covariance: torch.tensor (time, nnode, 6)
    position: Positions of particles (timesteps, nparticles, ndims)
    particle_types: Particles types with shape (nparticles)
    material_property: Friction angle normalized by tan() with shape (nparticles)
    n_particles_per_example
    nsteps: Number of steps.
    device: torch device.
    f_tensor: deformation gradient
    C: velocity reconstruction
  """
  covariance = covariance.half().to(device)

  position = position.half().to(device)
  particle_types = particle_types.to(device)  # 整数型は変換不要
  n_particles_per_example = n_particles_per_example.to(device)  # 整数型は変換不要
  if material_property is not None:
      material_property = material_property.half().to(device)
  if f_tensor is not None:
      f_tensor = f_tensor.half().to(device)
  if C is not None:
      C = C.half().to(device)

  # モデルも half() に変換
  simulator = simulator.half()

  # position
  initial_positions = position[:, :INPUT_SEQUENCE_LENGTH] # (nparticles, seq - seq_length, dim)
  ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:] #(nparticles, seq - inpu_seq_length, dim)

  current_positions = initial_positions

  # f_tensor
  current_f_tensors = None
  if f_tensor is not None:
    initial_f_tensors = f_tensor[:, INPUT_SEQUENCE_LENGTH-INPUT_TENSOR_SEQUENCE_LENGTH:INPUT_SEQUENCE_LENGTH]
    ground_truth_f_tensors = f_tensor[:, INPUT_SEQUENCE_LENGTH:]

    current_f_tensors = initial_f_tensors
    predictions_f_tensor = []

  # C
  current_C = None
  if C is not None:
    initial_C = C[:, INPUT_SEQUENCE_LENGTH-1:INPUT_SEQUENCE_LENGTH]
    ground_truth_C = C[:, INPUT_SEQUENCE_LENGTH:]

    current_C = initial_C

    predictions_C = []

  predictions = []


  kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)

  for step in tqdm(range(nsteps), total=nsteps):
    # Get next position with shape (nnodes, dim)
    with torch.cuda.amp.autocast():
      next_states = simulator.predict_positions(
          current_positions,
          nparticles_per_example = [n_particles_per_example],
          particle_types = particle_types,
          current_f_tensors = current_f_tensors,
          current_C = current_C,
          material_property = material_property
      )

    # position
    next_position = next_states["positions"]
    next_position_ground_truth = ground_truth_positions[:, step]
    kinematic_mask_pos = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
    next_position = torch.where(
        kinematic_mask_pos, next_position_ground_truth, next_position)
    predictions.append(next_position)

    # Shift `current_positions`, removing the oldest position in the sequence
    # and appending the next position at the end.
    current_positions = torch.cat(
        [current_positions[:, 1:], next_position[:, None, :]], dim=1)
    
    # f_tensor
    if f_tensor is not None:
      next_f_tensor = next_states["f_tensor"]
      next_f_tensor_ground_truth = ground_truth_f_tensors[:, step]
      kinematic_mask_f_tensor = kinematic_mask.bool()[:, None].expand(-1, current_f_tensors.shape[-1])
      next_f_tensor = torch.where(
        kinematic_mask_f_tensor, next_f_tensor_ground_truth, next_f_tensor)
      predictions_f_tensor.append(next_f_tensor)

      current_f_tensors = torch.cat(
        [current_f_tensors[:, 1:], next_f_tensor[:, None, :]], dim = 1)

    # C
    if C is not None:
      next_C = next_states["C"]
      next_C_ground_truth = ground_truth_C[:, step]
      kinematic_mask_C = kinematic_mask.bool()[:, None].expand(-1, current_C.shape[-1])
      next_C = torch.where(
        kinematic_mask_C, next_C_ground_truth, next_C)
      predictions_C.append(next_C)

      current_C = torch.cat(
        [current_C[:, 1:], next_C[:, None, :]], dim = 1)
      

  del simulator

  # Predictions with shape (time, nnodes, dim)
  predictions = torch.stack(predictions)
  ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

  loss = (predictions - ground_truth_positions) ** 2

  shape = (loss.shape[0], loss.shape[1], loss.shape[2] * loss.shape[2])
  loss_f = torch.zeros(shape)
  loss_C = torch.zeros(shape)

  ### xの予測
  predictions_x = torch.cat([position.permute(1, 0, 2)[:INPUT_SEQUENCE_LENGTH], predictions], dim=0)
  predictions_x = predictions_x.half()
  ###

  if f_tensor is not None:
    predictions_f_tensor = torch.stack(predictions_f_tensor)
    predictions_f_tensor = predictions_f_tensor.half()
    ground_truth_f_tensors = ground_truth_f_tensors.permute(1, 0, 2)

    loss_f = (predictions_f_tensor - ground_truth_f_tensors) ** 2

    # print(predictions_f_tensor)
    ###### cov, R
    ground_truth_covariance = covariance.clone().detach()

    predictions_covariance = repeat_to_match_length(
        ground_truth_covariance, INPUT_SEQUENCE_LENGTH
    ) 


    covariance33 = expand_covariance(covariance)[0] # (nnode, 3, 3)

    predictions_covariance = torch.cat([predictions_covariance[:INPUT_SEQUENCE_LENGTH], 
                                 flatten_covariance(
                                   compute_transformed_covariance(predictions_f_tensor, covariance33)
                                   )],dim = 0) # (time, nnode, 6)
    
    ground_truth_R = compute_R_from_F_pytorch(f_tensor.permute(1,0,2))
    predictions_R = torch.cat([ground_truth_R[:INPUT_SEQUENCE_LENGTH], 
                               compute_R_from_F_pytorch(predictions_f_tensor)], dim = 0) # (time, nnode,9)

    if ground_truth_covariance.shape[0] > 1:
      loss_covariance = (ground_truth_covariance[INPUT_SEQUENCE_LENGTH:] - predictions_covariance[INPUT_SEQUENCE_LENGTH:])**2
    else:
      loss_covariance = (ground_truth_covariance - predictions_covariance[INPUT_SEQUENCE_LENGTH:])**2

    loss_R = (ground_truth_R[INPUT_SEQUENCE_LENGTH:]- predictions_R[INPUT_SEQUENCE_LENGTH:])**2
    #######

  
  if C is not None:
    predictions_C = torch.stack(predictions_C)
    predictions_C = predictions_C.half()
    ground_truth_C = ground_truth_C.permute(1, 0, 2)

    loss_C = (predictions_C - ground_truth_C) ** 2

  output_dict = {
      'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
      'predicted_rollout': predictions.cpu().numpy(),
      'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
      'particle_types': particle_types.cpu().numpy(),
      'material_property': material_property.cpu().numpy() if material_property is not None else None
  }

  if f_tensor is not None:
    output_dict["initial_f_tensors"] = initial_f_tensors.permute(1, 0, 2).cpu().numpy()
    output_dict["predicted_rollout_f_tensor"] = predictions_f_tensor.cpu().numpy(),
    output_dict["ground_truth_rollout_f_tensor"] = ground_truth_f_tensors.cpu().numpy()

  if C is not None:
    output_dict["initial_C"] = initial_C.permute(1, 0, 2).cpu().numpy()
    output_dict["predicted_rollout_C"] = predictions_C.cpu().numpy()
    output_dict["ground_truth_rollout_C"] = ground_truth_C.cpu().numpy()
  
  if predictions_x is not None and predictions_x.dtype == torch.float16:
    predictions_x = predictions_x.float()
  if predictions_covariance is not None and predictions_covariance.dtype == torch.float16:
    predictions_covariance = predictions_covariance.float()
  if predictions_R is not None and predictions_R.dtype == torch.float16:
    predictions_R = predictions_R.float()

  return output_dict, loss, loss_f, loss_C, loss_covariance, loss_R, predictions_x ,predictions_covariance, predictions_R


def predict(FLAGS, device: str, covariance: torch.Tensor):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.
    covariance: torch.tensor (time, nnode, 6)

  """
  covariance.to(device)

  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path, "rollout")
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device, FLAGS.noise_tensor_std, FLAGS.noise_tensor_std)

  # Load simulator
  if os.path.exists(FLAGS.model_file):
    simulator.load(FLAGS.model_file)
  else:
    raise Exception(f"Model does not exist at {FLAGS.model_file}")

  simulator.to(device)
  simulator.eval()


  split = 'gaussian'

  # Get dataset
  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz", use_material_properties_list = use_material_properties_list, use_f_tensor = use_f_tensor, use_C_tensor=use_C_tensor)

  # See if our dataset has material property, C, f_tensor as feature
  material_property_as_feature = np.any(use_material_properties_list)
  C_as_feature = use_C_tensor
  f_tensor_as_feature = use_f_tensor

  eval_loss   = []
  eval_loss_f = []
  eval_loss_C = []
  eval_loss_cov=[]
  eval_loss_R = []
  with torch.no_grad():
    for example_i, features in enumerate(ds):
      print(f"processing example number {example_i}")
      positions = features["positions"].to(device)
      if metadata['sequence_length'] is not None:
        # If `sequence_length` is predefined in metadata,
        nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
      else:
        # If no predefined `sequence_length`, then get the sequence length
        sequence_length = positions.shape[1]
        nsteps = sequence_length - INPUT_SEQUENCE_LENGTH
      
      particle_type = features["particle_types"].to(device)
      n_particles_per_example = torch.tensor([int(features["n_particles_per_example"])], dtype=torch.int32).to(device)

      material_property = None
      if material_property_as_feature:
        material_property = features["material_properties"].to(device)
      
      f_tensor= None
      if f_tensor_as_feature:
        f_tensor = features["f_tensor"].to(device)
      
      C = None
      if C_as_feature:
        C = features["C"].to(device)
      

      # Predict example rollout
      example_rollout, loss, loss_f, loss_C, loss_covariance, loss_R, predictions_x,predictions_covariance, predictions_R = rollout(FLAGS,
                                                                                                                                              simulator,
                                      covariance,
                                      positions,
                                      particle_type,
                                      material_property,
                                      n_particles_per_example,
                                      nsteps,
                                      device,
                                      f_tensor,
                                      C)

      example_rollout['metadata'] = metadata
      print(f"Predicting example {example_i} loss: {loss.mean()} loss_f_tensor: {loss_f.mean()} loss_C: {loss_C.mean()}")
      eval_loss.append(torch.flatten(loss))
      eval_loss_f.append(torch.flatten(loss_f))
      eval_loss_C.append(torch.flatten(loss_C))
      eval_loss_cov.append(torch.flatten(loss_covariance))
      eval_loss_R.append(torch.flatten(loss_R))

      # Save rollout in testing
      example_rollout['metadata'] = metadata
      example_rollout['loss']   = loss.mean()
      example_rollout['loss_f'] = loss_f.mean()
      example_rollout['loss_C'] = loss_C.mean()
      example_rollout['loss_covariance'] = loss_covariance.mean()
      example_rollout['loss_R'] =loss_R.mean()
      filename = f'{FLAGS.output_filename}_ex{example_i}.pkl'
      filename = os.path.join(filename)
      with open(filename, 'wb') as f:
        pickle.dump(example_rollout, f)

  print(f"[Mean loss on rollout prediction] loss: {torch.mean(torch.cat(eval_loss)) } loss_f: {torch.mean(torch.cat(eval_loss_f))} loss_C: {torch.mean(torch.cat(eval_loss_C))} loss_covariance: {torch.mean(torch.cat(eval_loss_cov))} loss_R : {torch.mean(torch.cat(eval_loss_R))}")
  return predictions_x, predictions_covariance, predictions_R


def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)

def _get_simulator( # modify metadata
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: torch.device,
        f_tensor_diff_std: float = 0,
        C_diff_std: float = 0) -> learned_simulator.LearnedSimulator:
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    acc_noise_std: Acceleration noise std deviation.
    vel_noise_std: Velocity noise std deviation.
    device: PyTorch device 'cpu' or 'cuda'.
    f_tensor_diff_std: f_tensor_diff std deviation.
    C_diff_std: C_diff std deviation.
  """

  # Normalization stats
  normalization_stats = {
      'acceleration': {
          'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 +
                            acc_noise_std**2).to(device),
      },
      'velocity': {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                            vel_noise_std**2).to(device),
      },
      'f_tensor_diff': {
          'mean': torch.FloatTensor(metadata['f_tensor_diff_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['f_tensor_diff_std'])**2 +
                            f_tensor_diff_std**2).to(device),
      },
      'C_diff': {
          'mean': torch.FloatTensor(metadata['C_diff_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['C_diff_std'])**2 +
                            C_diff_std**2).to(device),
      },
  }

  # Get necessary parameters for loading simulator.
  particle_type_embedding_size=16
  material_dim = np.sum(use_material_properties_list)

  dim = metadata['dim']
  nnode_out = dim + (dim*dim) * use_f_tensor + (dim*dim) * use_C_tensor

  if "nnode_in" in metadata and "nedge_in" in metadata:
    nnode_in = metadata['nnode_in']
    nedge_in = metadata['nedge_in']
  else:
    # Given that there is no additional node feature (e.g., material_property) except for:
    # (position (dim), velocity (dim*6), particle_type (16)),
    nnode_in = dim * INPUT_SEQUENCE_LENGTH + (dim * dim * INPUT_TENSOR_SEQUENCE_LENGTH * use_f_tensor) + (dim * dim *use_C_tensor) + dim*2 + particle_type_embedding_size + material_dim
    nedge_in = dim + 1
    print(f"node feature dim: {nnode_in}, edge feature dim: {nedge_in}")

  # Init simulator.
  simulator = learned_simulator.LearnedSimulator(
      nnode_out=nnode_out,
      nnode_in=nnode_in,
      nedge_in=nedge_in,
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=mlp_hidden_dim,
      use_kan = use_KAN,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types = NUM_PARTICLE_TYPES,
      particle_type_embedding_size = particle_type_embedding_size,
      boundary_clamp_limit=metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0,
      device=device)

  return simulator

def main(_):
  """Evaluates the model.

  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device == torch.device('cuda'):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

  # Set device
  world_size = torch.cuda.device_count()
  if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
    device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
  #test code
  print(f"device is {device} world size is {world_size}")
  predict(device)


if __name__ == '__main__':
  app.run(main)
