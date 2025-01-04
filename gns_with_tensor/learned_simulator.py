import torch
import torch.nn as nn
import numpy as np
from gns_with_tensor import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict


class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
          self,
          nnode_out: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          boundaries: np.ndarray,
          normalization_stats: dict,
          nparticle_types: int,
          particle_type_embedding_size: int,
          boundary_clamp_limit: float = 1.0,
          device="cpu"
  ):
    """Initializes the model.

    Args:
      nnode_out: Number of node outputs
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      boundary_clamp_limit: a factor to enlarge connectivity radius used for computing
        normalized clipped distance in edge feature.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()
    self._boundaries = boundaries
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._nparticle_types = nparticle_types
    self._boundary_clamp_limit = boundary_clamp_limit

    # Particle type embedding has shape (9, 16)
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features = nnode_in,
        nnode_out_features = nnode_out,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

    self._device = device

  def forward(self):
    """Forward hook runs on class instantiation"""
    pass

  def _compute_graph_connectivity(
          self,
          node_features: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
    """Generate graph edges to all particles within a threshold radius

    Args:
      node_features: Node features with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      radius: Threshold to construct edges to all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """
    # Specify examples id for particles
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)

    # radius_graph accepts r < radius not r <= radius
    # A torch tensor list of source and target nodes with shape (2, nedges)
    edge_index = radius_graph(
        node_features, r=radius, batch=batch_ids, loop=add_self_edges, max_num_neighbors=128)

    # The flow direction when using in combination with message passing is
    # "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]

    return receivers, senders

  # modify (done)
  def _encoder_preprocessor(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          f_tensor_sequence: torch.tensor = None, 
          C_sequence: torch.tensor= None,
          material_property: torch.tensor = None):
    """Extracts important features from the position sequence. Returns a tuple
    of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
    edge_features (nparticles, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, input_seq_length, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
      f_tensor_sequence: A sequence of F tensor. Shape is (nparticles, input_tensor_seq_length, dim * dim)
      C_sequence: A sequence of C tensor. Shape is (nparticles, 1, dim * dim)
      material_property: Friction angle normalized by tan() with shape (nparticles, material_dim)
    """
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]  # (n_nodes, dim)
    
    if f_tensor_sequence is not None:
      most_recent_f_tensor = f_tensor_sequence[:, -1]  # (n_nodes, dim * dim)

    if C_sequence is not None:
      most_recent_C_tensor = C_sequence[:, -1]  # (n_nodes, dim * dim)

    # Get connectivity of the graph with shape of (nparticles, 2)
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    node_features = [most_recent_position] # position add

    if f_tensor_sequence is not None:
      node_features.append(most_recent_f_tensor)
    if C_sequence is not None:
      node_features.append(most_recent_C_tensor)

    if position_sequence.shape[1] > 1:
      velocity_sequence = time_diff(position_sequence)
      # Normalized velocity sequence, merging spatial an time axis.
      velocity_stats = self._normalization_stats["velocity"]
      normalized_velocity_sequence = (
          velocity_sequence - velocity_stats['mean']) / velocity_stats['std'] # (nparticles, seq-1, dim)
      flat_velocity_sequence = normalized_velocity_sequence.view(
          nparticles, -1) # (nparticles, (seq-1) * dim)
      # There are 5 previous steps, with dim 2
      # node_features shape (nparticles, 5 * 2 = 10)
      node_features.append(flat_velocity_sequence)
    
    if f_tensor_sequence is not None and f_tensor_sequence.shape[1] > 1:
      f_tensor_diff_sequence = time_diff(f_tensor_sequence)

      f_tensor_stats = self._normalization_stats["f_tensor_diff"]
      normalized_f_tensor_diff_sequence = (
          f_tensor_diff_sequence - f_tensor_stats['mean']) / f_tensor_stats['std']
      flat_f_tensor_sequence = normalized_f_tensor_diff_sequence.view(nparticles, -1)
      node_features.append(flat_f_tensor_sequence)

    # Normalized clipped distances to lower and upper boundaries.
    # boundaries are an array of shape [num_dimensions, 2], where the second
    # axis, provides the lower/upper boundaries.
    boundaries = torch.tensor(
        self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = (
        most_recent_position - boundaries[:, 0][None]) # (n_nodes, dim)
    distance_to_upper_boundary = (
        boundaries[:, 1][None] - most_recent_position) # (n_nodes, dim)
    distance_to_boundaries = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1) # (n_nodes, 2*dim)
    normalized_clipped_distance_to_boundaries = torch.clamp(
        distance_to_boundaries / self._connectivity_radius,
        -self._boundary_clamp_limit, self._boundary_clamp_limit)
    # 2D : The distance to 4 boundaries (top/bottom/left/right)
    # 3D : 6 boundaries
    # node_features shape (nparticles, 10+4)
    node_features.append(normalized_clipped_distance_to_boundaries)


    # Particle type
    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(
          particle_types)
      node_features.append(particle_type_embeddings)


    # Material property
    if material_property is not None:
        material_property = material_property #次のコードを削除->.view(nparticles, 1)
        node_features.append(material_property)

    # 以下変更なし 
    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    # with shape (nedges, 2)
    # normalized_relative_displacements = (
    #     torch.gather(most_recent_position, 0, senders) -
    #     torch.gather(most_recent_position, 0, receivers)
    # ) / self._connectivity_radius
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius

    # Add relative displacement between two particles as an edge feature
    # with shape (nparticles, ndim)
    edge_features.append(normalized_relative_displacements)

    # Add relative distance between 2 particles with shape (nparticles, 1)
    # Edge features has a final shape of (nparticles, ndim + 1)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))
  # modify(done)
  def _decoder_postprocessor(
          self,
          normalized_acceleration: torch.tensor,
          position_sequence: torch.tensor,
          f_tensor_sequence: torch.tensor = None,
          C_sequence: torch.tensor = None) -> torch.tensor:
    """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, outputDim).
      position_sequence: Position sequence of shape (nparticles, input_sequence_length, dim).
      f_tensor_sequence: deformation gradient sequence of shape (nparticles, input_tensor_seq_length, dim * dim)
      C_sequence: sequence of shape (nparticles, 1, dim * dim)

    Returns:
      dict
        positions: torch.tensor: New position of the particles. (nparticles, dim)
        f_tensor(optional): torch.tensor: New f_tensor of the particles. (nparticles, dim * dim)
        C(optional): torch.tensor: New C of the particles. (nparticles, dim * dim)

    """
    # Extract real acceleration values from normalized values
    acceleration_stats = self._normalization_stats["acceleration"]
       
    dim = position_sequence.shape[2]
    
    acceleration = (
        normalized_acceleration[:, :dim] * acceleration_stats['std']
    ) + acceleration_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1] # (nparticles, dim)
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    # TODO: Fix dt
    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    
    outputs = {
      "positions":new_position
    }

    start_id = dim

    if f_tensor_sequence is not None:
      f_tensor_stats = self._normalization_stats["f_tensor_diff"]
      f_tensor_diff = (
          normalized_acceleration[:, start_id : start_id + dim * dim] * f_tensor_stats["std"]
      ) + f_tensor_stats['mean']

      most_recent_f_tensor = f_tensor_sequence[:, -1]
      new_f_tensor = most_recent_f_tensor + f_tensor_diff
      outputs["f_tensor"] = new_f_tensor

      start_id += dim * dim
    
    if C_sequence is not None:
      C_stats = self._normalization_stats["C_diff"]
      C_diff = (
        normalized_acceleration[:, start_id : start_id + dim * dim] * C_stats["std"]
      ) + C_stats["mean"]

      most_recent_C_tensor = C_sequence[:, -1]
      new_C = most_recent_C_tensor + C_diff
      outputs["C"] = new_C

    return outputs
  
  # modify(done)
  def predict_positions(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          current_f_tensors: torch.tensor = None,
          current_C: torch.tensor = None,
          material_property: torch.tensor = None) -> torch.tensor:
    """Predict position based on acceleration.

    Args:
      current_positions: Current particle positions (nparticles, input_seq_length, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
      material_property: Friction angle normalized by tan() with shape (nparticles, material_dim)

      current_f_tensors: torch.tensor = None: Current deformation gradient tensor (nparticles, input_tensor_seq_length, dim * dim)
      current_C: torch.tensor = None: Current C tensor (nparticles, 1, dim * dim)

    Returns:
      dict
        positions: torch.tensor: New position of the particles.
        f_tensor(optional): torch.tensor: New f_tensor of the particles.
        C(optional): torch.tensor: New C of the particles.
    """
    if material_property is not None:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types, current_f_tensors, current_C, material_property)
    else:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types, current_f_tensors, current_C)
    predicted_normalized = self._encode_process_decode(
        node_features, edge_index, edge_features)
    next_states = self._decoder_postprocessor(
        predicted_normalized, current_positions, current_f_tensors, current_C)
    return next_states

  # modify
  def predict_accelerations(
          self,
          next_positions: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          next_f_tensor :torch.tensor = None,
          f_tensor_sequence_noise: torch.tensor = None,
          f_tensor_sequence:torch.tensor = None,
          next_C : torch.tensor = None,
          C_sequence_noise :torch.tensor = None,
          C_sequence :torch.tensor = None,
          material_property: torch.tensor = None):
    """Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, input_sequence_length, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

      next_f_tensor: Tensor of shape (nparticles_in_batch, dim * dim) with the
        f_tensor the model should output given the inputs.
      f_tensor_sequence_noise: Tensor of the same shape as `f_tensor_sequence`
        with the noise to apply to each particle.
      f_tensor_sequence: A sequence of particle positions. Shape is
        (nparticles, input_tensor_sequence_length, dim). Includes current + last 1 f_tensor.

      next_C: Tensor of shape (nparticles_in_batch, dim * dim) with the
        C the model should output given the inputs.
      C_sequence_noise: Tensor of the same shape as `C_sequence`
        with the noise to apply to each particle.
      C_sequence: A sequence of particle positions. Shape is
        (nparticles, 1, dim). Includes current

      material_property: Friction angle normalized by tan() with shape (nparticles).

    Returns:
      predict (dict):
        acc : normalized_acceleration (torch.tensor): Normalized acceleration.
        f_tensor_diff : normalized f_tensor difference 
        C_diff : normalized C_tensor difference
      target (dict):
        acc : normalized_acceleration (torch.tensor): Normalized acceleration.
        f_tensor_diff : normalized f_tensor difference 
        C_diff : normalized C_tensor difference

    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    noisy_f_tensor_sequence = None
    if f_tensor_sequence is not None:
        noisy_f_tensor_sequence = f_tensor_sequence + f_tensor_sequence_noise

    noisy_C_sequence = None
    if C_sequence is not None:
        noisy_C_sequence = C_sequence + C_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    if material_property is not None:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types, noisy_f_tensor_sequence, noisy_C_sequence, material_property)
    else:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types, noisy_f_tensor_sequence, noisy_C_sequence)
    predicted_normalized = self._encode_process_decode(
        node_features, edge_index, edge_features)
      
    dim = position_sequence.shape[2]
    predicted = {
      "acc": predicted_normalized[:, :dim]
    }
    start_id = dim
    if f_tensor_sequence is not None:
      predicted["f_tensor_diff"] = predicted_normalized[:, start_id : start_id + dim * dim]
      start_id += dim * dim
    if C_sequence is not None:
      predicted["C_diff"] = predicted_normalized[:, start_id : start_id + dim * dim]

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_positions + position_sequence_noise[:, -1]

    next_f_tensor_adjusted = None
    if f_tensor_sequence is not None:
      next_f_tensor_adjusted = next_f_tensor + f_tensor_sequence_noise[:, -1]

    next_C_adjusted = None
    if C_sequence is not None:
      next_C_adjusted = next_C + C_sequence_noise[:, -1]     

    target_normalized = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence,
        next_f_tensor_adjusted, noisy_f_tensor_sequence,
        next_C_adjusted, noisy_C_sequence,
        )
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted, target_normalized

  # modify
  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.tensor,
          position_sequence: torch.tensor,
          next_f_tensor: torch.tensor = None,
          f_tensor_sequence: torch.tensor = None,
          next_C: torch.tensor = None,
          C_sequence: torch.tensor = None
          ):
    """Inverse of `_decoder_postprocessor`.

    Args:
      next_position: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      
      next_f_tensor: Tensor of shape (nparticles_in_batch, dim) with the
        f_tensor the model should output given the inputs.
      f_tensor_sequence: A sequence of particle f_tensor. Shape is
        (nparticles, 2, dim). Includes current + last 1 positions.

      next_C: Tensor of shape (nparticles_in_batch, dim) with the
        C the model should output given the inputs.
      C_sequence: A sequence of particle C. Shape is
        (nparticles, 1, dim). Includes current.

    Returns:
      dict:
        acc : normalized_acceleration (torch.tensor): Normalized acceleration.
        f_tensor_diff : normalized f_tensor difference 
        C_diff : normalized C_tensor difference

    """


    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats['mean']) / acceleration_stats['std']

    outputs = {
      "acc": normalized_acceleration
    }

    if f_tensor_sequence is not None:
      previous_f_tensor = f_tensor_sequence[:, -1]
      f_tensor_stats = self._normalization_stats["f_tensor_diff"]
      f_tensor_diff = next_f_tensor - previous_f_tensor

      normalized_f_tensor_diff = (
          f_tensor_diff - f_tensor_stats["mean"]
      ) / f_tensor_stats['std']
      outputs["f_tensor_diff"] = normalized_f_tensor_diff
    
    if C_sequence is not None:
      previous_C = C_sequence[:, -1]
      C_stats = self._normalization_stats["C_diff"]
      C_diff = next_C - previous_C

      normalized_C_diff = (
        C_diff - C_stats["mean"]
      ) / C_stats["std"]
      outputs["C_diff"] = normalized_C_diff
    
    return outputs

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    """Load model state from file

    Args:
      path: Model path
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return position_sequence[:, 1:] - position_sequence[:, :-1]
