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

# pip install absl-py
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
from absl import flags
from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gns_with_tensor import learned_simulator
from gns_with_tensor import noise_utils
from gns_with_tensor import reading_utils
from gns_with_tensor import data_loader
from gns_with_tensor import distribute

# python3 train.py --mode train --data_path sample/

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_float('noise_tensor_std', 6.7e-9, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('output_filename', 'rollout', help='Base name for saving the rollout')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('validation_interval', None, help='Validation interval. Set `None` if validation loss is not needed')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
flags.DEFINE_integer("n_gpus", 1, help="The number of GPUs to utilize for training.")

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
INPUT_TENSOR_SEQUENCE_LENGTH = 2 # last 1 tensor difference f_tensor
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 7

loss_weight = {
  "acc": 1.,
  "f_tensor_diff": 1.,
  "C_diff": 1.,
}

use_material_properties_list = np.array([False]*15)
# use_material_properties_list[10] = True

use_f_tensor = True
use_C_tensor = True

def rollout(
        simulator: learned_simulator.LearnedSimulator,
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
    position: Positions of particles (timesteps, nparticles, ndims)
    particle_types: Particles types with shape (nparticles)
    material_property: Friction angle normalized by tan() with shape (nparticles)
    n_particles_per_example
    nsteps: Number of steps.
    device: torch device.
    f_tensor: deformation gradient
    C: velocity reconstruction
  """

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

  for step in tqdm(range(nsteps), total=nsteps):
    # Get next position with shape (nnodes, dim)
    next_states = simulator.predict_positions(
        current_positions,
        nparticles_per_example = [n_particles_per_example],
        particle_types = particle_types,
        current_f_tensors = current_f_tensors,
        current_C = current_C,
        material_property = material_property
    )
    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)

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
      


  # Predictions with shape (time, nnodes, dim)
  predictions = torch.stack(predictions)
  ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

  loss = (predictions - ground_truth_positions) ** 2

  shape = (loss.shape[0], loss.shape[1], loss.shape[2] * loss_shape[2])
  loss_f = torch.zeros(shape)
  loss_C = torch.zeros(shape)

  if f_tensor is not None:
    predictions_f_tensor = torch.stack(predictions_f_tensor)
    ground_truth_f_tensors = ground_truth_f_tensors.permute(1, 0, 2)

    loss_f = (predictions_f_tensor - ground_truth_f_tensors) ** 2
  
  if C is not None:
    predictions_C = torch.stack(predictions_C)
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

  return output_dict, loss, loss_f, loss_C


def predict(device: str):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path, "rollout")
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device, FLAGS.noise_tensor_std, FLAGS.noise_tensor_std)

  # Load simulator
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")

  simulator.to(device)
  simulator.eval()

  # Output path
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  # Use `valid`` set for eval mode if not use `test`
  split = 'test' if (FLAGS.mode == 'rollout' or (not os.path.isfile("{FLAGS.data_path}valid.npz"))) else 'valid'

  # Get dataset
  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz", use_material_properties_list = use_material_properties_list, use_f_tensor = use_f_tensor, use_C_tensor=use_C_tensor)

  # See if our dataset has material property, C, f_tensor as feature
  material_property_as_feature = True if "material_properties" in ds.dataset._data[0] else False
  C_as_feature = True if "C" in ds.dataset._data[0] else False
  f_tensor_as_feature = True if "f_tensor" in ds.dataset._data[0] else False

  eval_loss   = []
  eval_loss_f = []
  eval_loss_C = []
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

      if material_property_as_feature:
        material_property = features["material_properties"].to(device)
      
      f_tensor= None
      if f_tensor_as_feature:
        f_tensor = features["f_tensor"].to(device)
      
      C = None
      if C_as_feature:
        C = features["C"].to(device)
      

      # Predict example rollout
      example_rollout, loss, loss_f, loss_C = rollout(simulator,
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

      # Save rollout in testing
      if FLAGS.mode == 'rollout':
        example_rollout['metadata'] = metadata
        example_rollout['loss']   = loss.mean()
        example_rollout['loss_f'] = loss_f.mean()
        example_rollout['loss_C'] = loss_C.mean()
        filename = f'{FLAGS.output_filename}_ex{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print(f"[Mean loss on rollout prediction] loss: {torch.mean(torch.cat(eval_loss)) } loss_f: {torch.mean(torch.cat(eval_loss_f))} loss_C: {torch.mean(torch.cat(eval_loss_C))}")


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

def acceleration_loss(pred_acc, target_acc, non_kinematic_mask):
  """
  Compute the loss between predicted and target accelerations.

  Args:
    pred_acc: Predicted accelerations.
    target_acc: Target accelerations.
    non_kinematic_mask: Mask for kinematic particles.
  """
  loss = (pred_acc - target_acc) ** 2 # (nparticles, dim)
  loss = loss.sum(dim=-1) # (nparticles,)
  num_non_kinematic = non_kinematic_mask.sum()
  loss = torch.where(non_kinematic_mask.bool(),
                    loss, torch.zeros_like(loss))
  loss = loss.sum() / num_non_kinematic
  return loss # divide non-kinematic particles

def save_model_and_train_state(rank, device, simulator, flags, step, epoch, optimizer,
                                train_loss, valid_loss, train_loss_hist, valid_loss_hist):
  """Save model state
  
  Args:
    rank: local rank
    device: torch device type
    simulator: Trained simulator if not will undergo training.
    flags: flags
    step: step
    epoch: epoch
    optimizer: optimizer
    train_loss: training loss at current step
    valid_loss: validation loss at current step
    train_loss_hist: training loss history at each epoch
    valid_loss_hist: validation loss history at each epoch
  """
  if rank == 0 or device == torch.device("cpu"):
      if device == torch.device("cpu"):
          simulator.save(flags["model_path"] + 'model-' + str(step) + '.pt')
      else:
          simulator.module.save(flags["model_path"] + 'model-' + str(step) + '.pt')

      train_state = dict(optimizer_state=optimizer.state_dict(),
                          global_train_state={
                            "step": step, 
                            "epoch": epoch,
                            "train_loss": train_loss["total"],
                            "valid_loss": valid_loss["total"],
                            },
                          tot_loss_history={"train": train_loss_hist["total"], "valid": valid_loss_hist["total"]},
                          acc_loss_history={"train": train_loss_hist["acc"], "valid": valid_loss_hist["acc"]},
                          f_tensor_diff_loss_history={"train": train_loss_hist["f_tensor_diff"], "valid": valid_loss_hist["f_tensor_diff"]},
                          C_diff_loss_history={"train": train_loss_hist["C_diff"], "valid": valid_loss_hist["C_diff"]}
                          )
      torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

def save_loss_history_plots(train_loss_hist, valid_loss_hist, flags):
    """
    Save loss history plots for training and validation losses.
    
    Args:
        train_loss_hist (dict): Training loss history, with keys like "total", "acc", etc.
        valid_loss_hist (dict): Validation loss history, with the same structure as train_loss_hist.
        flags (dict): Dictionary containing configuration, including "model_path".
    """
    # Ensure the output directory exists
    output_path = flags["model_path"]
    os.makedirs(output_path, exist_ok=True)
    
    # Iterate over each key in the loss history
    for key in train_loss_hist.keys():
        # Extract epochs and loss values for training and validation
        train_epochs, train_losses = zip(*train_loss_hist[key])
        valid_epochs, valid_losses = zip(*valid_loss_hist[key])
        
        # Create the plot
        plt.figure()
        plt.plot(train_epochs, train_losses, label='Train', marker='o')
        plt.plot(valid_epochs, valid_losses, label='Validation', marker='x')
        plt.title(f'{key.capitalize()} Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        
        # Save the plot
        file_path = os.path.join(output_path, f'{key}_loss_history.png')
        plt.savefig(file_path)
        plt.close()

def train(rank, flags, world_size, device):
  """Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
    device: torch device type
  """
  if device == torch.device("cuda"):
    distribute.setup(rank, world_size, device)
    device_id = rank
  else:
    device_id = device

  # Read metadata
  metadata = reading_utils.read_metadata(flags["data_path"], "train")

  # Get simulator and optimizer
  if device == torch.device("cuda"):
    serial_simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], rank, flags["noise_tensor_std"], flags["noise_tensor_std"])
    simulator = DDP(serial_simulator.to(rank), device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*world_size)
  else:
    simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device, flags["noise_tensor_std"], flags["noise_tensor_std"])
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"] * world_size)

  # Initialize training state
  step = 0
  epoch = 0
  steps_per_epoch = 0

  train_loss = {"total": 0, "acc": 0, "f_tensor_diff": 0, "C_diff": 0}
  valid_loss = {"total": 0, "acc": 0, "f_tensor_diff": 0, "C_diff": 0}

  epoch_train_loss = {"total": 0, "acc": 0, "f_tensor_diff": 0, "C_diff": 0}
  epoch_valid_loss = {"total": 0, "acc": 0, "f_tensor_diff": 0, "C_diff": 0}

  train_loss_hist = {"total": [], "acc": [], "f_tensor_diff": [], "C_diff": []}
  valid_loss_hist = {"total": [], "acc": [], "f_tensor_diff": [], "C_diff": []}

  # If model_path does exist and model_file and train_state_file exist continue training.
  if flags["model_file"] is not None:

    if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
      # find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f'{flags["model_path"]}*model*pt')
      max_model_number = 0
      expr = re.compile(".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # reset names to point to the latest.
      flags["model_file"] = f"model-{max_model_number}.pt"
      flags["train_state_file"] = f"train_state-{max_model_number}.pt"

    if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
      # load model
      if device == torch.device("cuda"):
        simulator.module.load(flags["model_path"] + flags["model_file"])
      else:
        simulator.load(flags["model_path"] + flags["model_file"])

      # load train state
      train_state = torch.load(flags["model_path"] + flags["train_state_file"])
      
      # set optimizer state
      optimizer = torch.optim.Adam(
        simulator.module.parameters() if device == torch.device("cuda") else simulator.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, device_id)
      
      # set global train state
      step = train_state["global_train_state"]["step"]
      epoch = train_state["global_train_state"]["epoch"]

      train_loss_hist["total"] = train_state["tot_loss_history"]["train"]
      valid_loss_hist["total"] = train_state["tot_loss_history"]["valid"]

      train_loss_hist["acc"] = train_state["acc_loss_history"]["train"]
      valid_loss_hist["acc"] = train_state["acc_loss_history"]["valid"]

      train_loss_hist["f_tensor_diff"] = train_state["f_tensor_diff_loss_history"]["train"]
      valid_loss_hist["f_tensor_diff"] = train_state["f_tensor_diff_loss_history"]["valid"]

      train_loss_hist["C_diff"] = train_state["C_diff_loss_history"]["train"]
      valid_loss_hist["C_diff"] = train_state["C_diff_loss_history"]["valid"]

    else:
      msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
      raise FileNotFoundError(msg)

  simulator.train()
  simulator.to(device_id)

  # Get data loader
  get_data_loader = (
    distribute.get_data_distributed_dataloader_by_samples
    if device == torch.device("cuda")
    else data_loader.get_data_loader_by_samples
  )

  # Load training data
  dl = get_data_loader(
      path=f'{flags["data_path"]}train.npz',
      input_length_sequence=INPUT_SEQUENCE_LENGTH,
      input_length_tensor_sequence = INPUT_TENSOR_SEQUENCE_LENGTH,
      use_material_properties_list = use_material_properties_list,
      use_C_tensor = use_C_tensor,
      use_f_tensor = use_f_tensor,
      batch_size=flags["batch_size"],
  )
  n_features = len(dl.dataset._data[0])

  # Load validation data
  if flags["validation_interval"] is not None:
      dl_valid = get_data_loader(
          path=f'{flags["data_path"]}valid.npz',
          input_length_sequence=INPUT_SEQUENCE_LENGTH,
          input_length_tensor_sequence = INPUT_TENSOR_SEQUENCE_LENGTH,
          use_material_properties_list = use_material_properties_list,
          use_C_tensor = use_C_tensor,
          use_f_tensor = use_f_tensor,
          batch_size=flags["batch_size"],
      )
      if len(dl_valid.dataset._data[0]) != n_features:
          raise ValueError(
              f"`n_features` of `valid.npz` and `train.npz` should be the same"
          )

  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")

  try:
    while step < flags["ntraining_steps"]:
      if device == torch.device("cuda"):
        torch.distributed.barrier()

      for example in dl:  
        steps_per_epoch += 1
        # ((position, particle_type, material_property, n_particles_per_example), labels) are in dl
        position = example["positions"].to(device_id)
        particle_type = example["particle_types"].to(device_id)
        n_particles_per_example = example["n_particles_per_example"].to(device_id)
        
        labels = example["label"].to(device_id)

        C = None
        label_C = None
        if "C" in example:
          C = example["C"].to(device_id)
          label_C = example["label_C"].to(device_id)

        f_tensor = None
        label_f_tensor = None
        if "f_tensor" in example:
          f_tensor = example["f_tensor"].to(device_id)
          label_f_tensor = example["label_f_tensor"].to(device_id)
        
        material_properties = None
        if "material_properties" in example:
          material_properties = example["material_properties"].to(device_id)

        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device_id)

        # Sample the noise to add to the inputs to the model during training.
        sampled_noise_pos = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=flags["noise_std"]).to(device_id)
        sampled_noise_pos *= non_kinematic_mask.view(-1, 1, 1)

        # f_tensor
        sampled_noise_f = None
        if "f_tensor" in example:
          sampled_noise_f = noise_utils.get_random_walk_noise_for_position_sequence(f_tensor, noise_std_last_step=flags["noise_tensor_std"]).to(device_id)
          sampled_noise_f *= non_kinematic_mask.view(-1, 1, 1)

        # C_tensor
        sampled_noise_C = None        
        if "C" in example:
          sampled_noise_C = noise_utils.get_random_walk_noise_for_position_sequence(C, noise_std_last_step=flags["noise_tensor_std"]).to(device_id)
          sampled_noise_C *= non_kinematic_mask.view(-1, 1, 1)

        # Get the predictions and target accelerations
        device_or_rank = rank if device == torch.device("cuda") else device
        pred, target = (simulator.module.predict_accelerations if device == torch.device("cuda") else simulator.predict_accelerations)(
            next_positions=labels.to(device_or_rank),
            position_sequence_noise=sampled_noise_pos.to(device_or_rank),
            position_sequence=position.to(device_or_rank),
            nparticles_per_example=n_particles_per_example.to(device_or_rank),
            particle_types=particle_type.to(device_or_rank),
            next_f_tensor = label_f_tensor.to(device_or_rank) if label_f_tensor is not None else None,
            f_tensor_sequence_noise = sampled_noise_f.to(device_or_rank) if sampled_noise_f is not None else None,
            f_tensor_sequence = f_tensor.to(device_or_rank) if f_tensor is not None else None,
            next_C = label_C.to(device_or_rank) if label_C is not None else None,
            C_sequence_noise = sampled_noise_C.to(device_or_rank) if sampled_noise_C is not None else None,
            C_sequence = C.to(device_or_rank) if C is not None else None,
            material_property=material_properties.to(device_or_rank) if material_properties is not None else None
        )
        
        # Validation
        if flags["validation_interval"] is not None:
          sampled_valid_example = next(iter(dl_valid))
          if step > 0 and step % flags["validation_interval"] == 0:
              tot_valid_loss, components = validation(
                simulator, sampled_valid_example, flags, rank, device_id)
              print(f"Validation loss at {step}: {tot_valid_loss.item()}")
              valid_loss["total"] = tot_valid_loss.item()
              for key, value in components.items():
                print(f"({key}_loss : {value}")
                valid_loss[key] = value

        # Calculate the loss and mask out loss on kinematic particles
        loss_acc = loss_weight["acc"] * acceleration_loss(pred["acc"], target["acc"], non_kinematic_mask)
        loss = loss_acc
        train_loss["acc"] = loss_acc.item()

        if "f_tensor" in example:
          loss_f_tensor_diff = loss_weight["f_tensor_diff"] * acceleration_loss(pred["f_tensor_diff"], target["f_tensor_diff"], non_kinematic_mask)
          train_loss["f_tensor_diff"] = loss_f_tensor_diff.item()

          loss += loss_f_tensor_diff

        if "C" in example:
          loss_C_diff = loss_weight["C_diff"] * acceleration_loss(pred["C_diff"], target["C_diff"], non_kinematic_mask)
          train_loss["C_diff"] = loss_C_diff.item()

          loss += loss_C_diff

        train_loss["total"] = loss.item()
        for key in train_loss.keys():
          epoch_train_loss[key] += train_loss[key]

        # Computes the gradient of loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
        for param in optimizer.param_groups:
          param['lr'] = lr_new
     
        print(f'rank = {rank}, epoch = {epoch}, step = {step}/{flags["ntraining_steps"]}, loss = {train_loss["total"]}', flush=True)

        # Save model state
        if rank == 0 or device == torch.device("cpu"):
          if step % flags["nsave_steps"] == 0:
            save_model_and_train_state(rank, device, simulator, flags, step, epoch, 
                                       optimizer, train_loss, valid_loss, train_loss_hist, valid_loss_hist)

        step += 1
        if step >= flags["ntraining_steps"]:
            break

      # Epoch level statistics
      # Training loss at epoch
      for key in epoch_train_loss.keys():
        epoch_train_loss[key] /= steps_per_epoch
        epoch_train_loss[key] = torch.tensor([epoch_train_loss[key]]).to(device_id)
        if device == torch.device("cuda"):
          torch.distributed.reduce(epoch_train_loss[key], dst=0, op=torch.distributed.ReduceOp.SUM)
          epoch_train_loss[key] /= world_size

        train_loss_hist[key].append((epoch, epoch_train_loss[key].item()))

      # Validation loss at epoch
      if flags["validation_interval"] is not None:
        sampled_valid_example = next(iter(dl_valid))
        tot_epoch_valid_loss, components= validation(
                simulator, sampled_valid_example, n_features, flags, rank, device_id)
        if device == torch.device("cuda"):
          torch.distributed.reduce(tot_epoch_valid_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
          tot_epoch_valid_loss /= world_size

        epoch_valid_loss["total"] = tot_epoch_valid_loss.item()
        for key, value in components.items():
            epoch_valid_loss[key] = value
            valid_loss_hist[key].append((epoch,value))
        valid_loss_hist["total"].append((epoch, tot_epoch_valid_loss.item()))

      # Print epoch statistics
      if rank == 0 or device == torch.device("cpu"):
        print(f'Epoch {epoch}, training loss: {epoch_train_loss["total"].item()}')
        if flags["validation_interval"] is not None:
          print(f'Epoch {epoch}, validation loss: {epoch_valid_loss["total"].item()}')
      
      # Reset epoch training loss
      epoch_train_loss = {"total": 0, "acc": 0, "f_tensor_diff": 0, "C_diff": 0}
      if steps_per_epoch >= len(dl):
        epoch += 1
      steps_per_epoch = 0
      
      if step >= flags["ntraining_steps"]:
        break 
      
  except KeyboardInterrupt:
    pass

  # Save model state on keyboard interrupt
  save_model_and_train_state(rank, device, simulator, flags, step, epoch, optimizer, train_loss, valid_loss, train_loss_hist, valid_loss_hist)

  if torch.cuda.is_available():
    distribute.cleanup()


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
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types = NUM_PARTICLE_TYPES,
      particle_type_embedding_size = particle_type_embedding_size,
      boundary_clamp_limit=metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0,
      device=device)

  return simulator

def validation(
        simulator,
        example,
        flags,
        rank,
        device_id):

  position = example["positions"].to(device_id)
  particle_type = example["particle_types"].to(device_id)
  n_particles_per_example = example["n_particles_per_example"].to(device_id)
  labels = example["label"].to(device_id)

  material_property = None
  if "material_properties" in example:
    material_property = example["material_properties"].to(device_id)

  f_tensor = None
  label_f_tensor = None
  if "f_tensor" in example:
    f_tensor = example["f_tensor"].to(device_id)
    label_f_tensor = example["label_f_tensor"].to(device_id)
  
  C = None
  label_C = None
  if "C" in example:
    C = example["C"].to(device_id)
    label_C = example["label_C"].to(device_id)
  
  non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device_id)
  # Sample the noise to add to the inputs.
  sampled_noise_pos = noise_utils.get_random_walk_noise_for_position_sequence(
    position, noise_std_last_step=flags["noise_std"]).to(device_id)
  sampled_noise_pos *= non_kinematic_mask.view(-1, 1, 1)

   # f_tensor
  sampled_noise_f = None
  if "f_tensor" in example:
    sampled_noise_f = noise_utils.get_random_walk_noise_for_position_sequence(
      f_tensor, noise_std_last_step = flags["noise_tensor_std"]).to(device_id)
    sampled_noise_f *= non_kinematic_mask.view(-1, 1, 1)

  # C_tensor
  sampled_noise_C = None        
  if "C" in example:
    sampled_noise_C = noise_utils.get_random_walk_noise_for_position_sequence(
      C, noise_std_last_step = flags["noise_tensor_std"]).to(device_id)
    sampled_noise_C *= non_kinematic_mask.view(-1, 1, 1)

  # Do evaluation for the validation data
  device_or_rank = rank if isinstance(device_id, int) else device_id
  # Select the appropriate prediction function
  predict_accelerations = simulator.module.predict_accelerations if isinstance(device_id, int) else simulator.predict_accelerations
  # Get the predictions and target accelerations
  with torch.no_grad():
      pred, target = predict_accelerations(
          next_positions=labels.to(device_or_rank),
          position_sequence_noise=sampled_noise_pos.to(device_or_rank),
          position_sequence=position.to(device_or_rank),
          nparticles_per_example=n_particles_per_example.to(device_or_rank),
          particle_types=particle_type.to(device_or_rank),
          next_f_tensor = label_f_tensor.to(device_or_rank) if label_f_tensor is not None else None,
          f_tensor_sequence_noise = sampled_noise_f.to(device_or_rank) if sampled_noise_f is not None else None,
          f_tensor_sequence = f_tensor.to(device_or_rank) if f_tensor is not None else None,
          next_C = label_C.to(device_or_rank) if label_C is not None else None,
          C_sequence_noise = sampled_noise_C.to(device_or_rank) if sampled_noise_C is not None else None,
          C_sequence = C.to(device_or_rank) if C is not None else None,
          material_property=material_property.to(device_or_rank) if material_property is not None else None, 
      )
  device = pred["acc"].device
  # Compute loss
  loss_components = {
    "acc": torch.zeros(1, device=device),
    "f_tensor_diff": torch.zeros(1, device=device),
    "C_diff": torch.zeros(1, device=device)
  }

  loss_components["acc"] = loss_weight["acc"] * acceleration_loss(pred["acc"], target["acc"], non_kinematic_mask)
  
  if "f_tensor" in example:
    loss_components["f_tensor_diff"] = loss_weight["f_tensor_diff"] * acceleration_loss(pred["f_tensor_diff"], target["f_tensor_diff"], non_kinematic_mask)
  if "C" in example:
    loss_components["C_diff"] = loss_weight["C_diff"] * acceleration_loss(pred["C_diff"], target["C_diff"], non_kinematic_mask)
  # Total loss
  total_loss = sum(loss_components.values())

  for key, value in loss_components.items():
    loss_components[key] = value.item()
  return total_loss, loss_components


def main(_):
  """Train or evaluates the model.

  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device == torch.device('cuda'):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

  myflags = reading_utils.flags_to_dict(FLAGS)

  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)

    # Train on gpu 
    if device == torch.device('cuda'):
      available_gpus = torch.cuda.device_count()
      print(f"Available GPUs = {available_gpus}")

      # Set the number of GPUs based on availability and the specified number
      if FLAGS.n_gpus is None or FLAGS.n_gpus > available_gpus:
        world_size = available_gpus
        if FLAGS.n_gpus is not None:
          print(f"Warning: The number of GPUs specified ({FLAGS.n_gpus}) exceeds the available GPUs ({available_gpus})")
      else:
        world_size = FLAGS.n_gpus

      # Print the status of GPU usage
      print(f"Using {world_size}/{available_gpus} GPUs")

      # Spawn training to GPUs
      distribute.spawn_train(train, myflags, world_size, device)

    # Train on cpu  
    else:
      rank = None
      world_size = 1
      train(rank, myflags, world_size, device)

  elif FLAGS.mode in ['valid', 'rollout']:
    # Set device
    world_size = torch.cuda.device_count()
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    #test code
    print(f"device is {device} world size is {world_size}")
    predict(device)


if __name__ == '__main__':
  app.run(main)
