import torch
import numpy as np


def load_npz_data(path):
    """Load data stored in npz format.

    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.

    Returns:
        data (list): List of trajectory(dict).
    """
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = []
            for _, item in data_file.items():
                if isinstance(item, np.ndarray) and item.ndim == 0:
                    item = item.item()  # Unpack the 0-d array
                if isinstance(item, dict):
                    data.append(item)  
                elif isinstance(item, np.ndarray) and item.dtype == object:
                    for nested_item in item:
                        if isinstance(nested_item, dict):
                            data.append(nested_item)
                        else:
                            raise ValueError(f"Unexpected type in ndarray: {type(nested_item)}")
                else:
                    raise ValueError(f"Unexpected type in npz file: {type(item)}")
    return data


class SamplesDataset(torch.utils.data.Dataset):
    """Dataset of samples of trajectories.
    
    Each sample is a tuple of the form (positions, particle_type, material_properties, C, f_tensor).

    positions : numpy array of shape (sequence_length, n_particles, dimension).
    particle_type : numpy array of shape (n_particles, dtype = int)
    material_properties : numpy array of shape (15, dtype = float)

    [Added]
    C : numpy array of shape (sequence_length, n_particles, dimension * dimension)
    f_tensor : numpy array of shape (sequence_length, n_particles, dimension * dimension)

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input Position sequence
        
        [Added]
        input_length_tensor_sequence(int): Length of input tensor sequence
        use_material_properties_list(np.array, dtype = bool): Array of length 15 storing the flag whether to use property or not
        use_f_tensor (bool): Whether to use f_tensor
        use_C_tensor (bool): Whether to use C_tensor

    Attributes:
        _data (list): List of tuples of the form (positions, particle_type).
        _dimension (int): Dimension of the data.
        _input_length_sequence (int): Length of input sequence.
        _input_length_tensor_sequence (int): Length of input tensor sequence
        _data_lengths (list): List of lengths of trajectories in the dataset.
        _length (int): Total number of samples in the dataset.
        _precompute_cumlengths (np.array): Precomputed cumulative lengths of trajectories in the dataset.
    """

    def __init__(self, path, input_length_sequence = 6, input_length_tensor_sequence = 2, use_material_properties_list = np.array([False]*15, dtype=bool), use_f_tensor = False, use_C_tensor = False):
        super().__init__()
        if input_length_sequence < input_length_tensor_sequence:
            raise ValueError(f"input_length_sequence ({input_length_sequence}) must be greater than or equal to input_length_tensor_sequence ({input_length_tensor_sequence}).")

        if len(use_material_properties_list)!=15:
            raise ValueError(f"The length of use_material_properties_list must be 15, but got {len(use_material_properties_list)}.")
        if not isinstance(use_material_properties_list, np.ndarray):
            raise TypeError("use_material_properties_list must be a numpy array.")

        self._data = load_npz_data(path)
        
        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0]["positions"].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._input_length_tensor_sequence = input_length_tensor_sequence
        self._use_material_properties_list = use_material_properties_list
        
        self._material_property_as_feature =  np.any(use_material_properties_list)
        self._C_as_feature = use_C_tensor
        self._f_tensor_as_feature = use_f_tensor

        # trajectoryごとの系列長
        self._data_lengths = [x["positions"].shape[0] - self._input_length_sequence for x in self._data]
        # 合計の系列長
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__

        if not self._data_lengths:
            raise ValueError("The dataset is empty or improperly formatted.")
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths) + 1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        """Return length of dataset.
        
        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.
        
        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).
        """
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx - 1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)

        # Prepare training data.
        positions = self._data[trajectory_idx]["positions"][time_idx - self._input_length_sequence:time_idx]
        positions = np.transpose(positions, (1, 0, 2))  # nparticles, input_sequence_length, dimension
        particle_type = self._data[trajectory_idx]["particle_types"]

        n_particles_per_example = positions.shape[0] # 粒子数

        if self._material_property_as_feature:
            material_property = self._data[trajectory_idx]["material_properties"]
            material_property = material_property[self._use_material_properties_list] # filter
            material_property = np.tile(material_property, (n_particles_per_example, 1))

        if self._C_as_feature:
            C = self._data[trajectory_idx]["C"][time_idx - 1:time_idx]
            C = np.transpose(C, (1, 0, 2))

        if self._f_tensor_as_feature:
            f_tensor = self._data[trajectory_idx]["f_tensor"][time_idx - self._input_length_tensor_sequence:time_idx]
            f_tensor = np.transpose(f_tensor, (1, 0, 2))

        label = self._data[trajectory_idx]["positions"][time_idx]

        training_example = {
            "positions": positions, # (nparticles, input_sequence_length, dimension)
            "particle_types": particle_type, # (nparticles, )
            "n_particles_per_example": n_particles_per_example,#(1,)
            "label": label, # (nparticles, dimension)
        }

        if self._material_property_as_feature:
            training_example["material_properties"] = material_property  # (nparticles, attributes)
        if self._C_as_feature:
            training_example["C"] = C  # (nparticles, 1, dimension * dimension)
            label_C = self._data[trajectory_idx]["C"][time_idx]

            training_example["label_C"] = label_C

        if self._f_tensor_as_feature:
            training_example["f_tensor"] = f_tensor # (nparticles, input_tensor_sequence_length, dimension * dimension)
            label_f_tensor = self._data[trajectory_idx]["f_tensor"][time_idx]

            training_example["label_f_tensor"] = label_f_tensor

        return training_example


def collate_fn(data):
    """Collate function for SamplesDataset.
    positions : (sequence_length, n_particles, dimension)

    Args:
        data (list): List of dict {positions, particle_types, n_particles_per_example, label, ....}.
        [Added]
        dict:
            positions: np.array((nparticles, input_sequence_length, dimension))
            particle_types: np.array((nparticles, ))
            n_particle_per_example: int 
            label: np.array(((nparticles, dimension)))
            material_properties: np.array((nparticles, attributes)) # Optional
            C: np.array((nparticles, 1, dimension * dimension)) # Optional
            label_C: np.array((nparticles, dimension * dimension)) # Optional
            f_tensor: np.array((nparticles, input_tensor_sequence_length, dimension * dimension)) # Optional
            label_f_tensor: np.array((nparticles, dimension * dimension)) # Optional

    Returns:
        [Added]
        dict: dict of the form {positions, particle_types, n_particles_per_example, label, ....}.   

            positions: torch.Tensor((batch_size * n_particles, input_sequence_length, dimension))
            particle_types: torch.Tensor((batch_size * n_particles,))
            n_particles_per_example: torch.Tensor((batch_size,))
            label: torch.Tensor((batch_size * n_particles, dimension))
            material_properties: torch.Tensor((batch_size * n_particles, attributes))  # Optional
            C: torch.Tensor((batch_size * n_particles, 1, dimension * dimension))       # Optional
            label_C: torch.Tensor((batch_size * n_particles, dimension * dimension))    # Optional
            f_tensor: torch.Tensor((batch_size * n_particles, input_tensor_sequence_length, dimension * dimension))  # Optional
            label_f_tensor: torch.Tensor((batch_size * n_particles, dimension * dimension))  # Optional
    """
    material_property_as_feature = True if "material_properties" in data[0] else False
    C_as_feature = True if "C" in data[0] else False
    f_tensor_as_feature = True if "f_tensor" in data[0] else False
    
    position_list = []
    particle_type_list = []
    if C_as_feature:
        C_list = []
        label_C = []
    if f_tensor_as_feature:
        f_tensor_list = []
        label_f_tensor = []
    if material_property_as_feature:
        material_property_list = []
    n_particles_per_example_list = []
    label_list = []

    for comp in data:
        position_list.append(comp["positions"])
        particle_type_list.append(comp["particle_types"])
        n_particles_per_example_list.append(comp["n_particles_per_example"])
        label_list.append(comp["label"])

        if material_property_as_feature:
            material_property_list.append(comp["material_properties"])
        if C_as_feature:
            C_list.append(comp["C"])
            label_C.append(comp["label_C"])
        if f_tensor_as_feature:
            f_tensor_list.append(comp["f_tensor"])
            label_f_tensor.append(comp["label_f_tensor"])
    
    collated_data = {
        "positions": torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
        "particle_types": torch.tensor(np.concatenate(particle_type_list)).contiguous(), # 1D array 
        "n_particles_per_example": torch.tensor(n_particles_per_example_list).contiguous(),
        "label": torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
    }

    if material_property_as_feature:
        collated_data["material_properties"] = torch.tensor(np.concatenate(material_property_list)).to(torch.float32).contiguous()
    if C_as_feature:
        collated_data["C"] = torch.tensor(np.vstack(C_list)).to(torch.float32).contiguous()
        collated_data["label_C"] = torch.tensor(np.vstack(label_C)).to(torch.float32).contiguous()

    if f_tensor_as_feature:
        collated_data["f_tensor"] = torch.tensor(np.vstack(f_tensor_list)).to(torch.float32).contiguous()
        collated_data["label_f_tensor"] = torch.tensor(np.vstack(label_f_tensor)).to(torch.float32).contiguous()

    return collated_data


class TrajectoriesDataset(torch.utils.data.Dataset):
    """Dataset of trajectories.

    Each trajectory is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    """

    def __init__(self, path, use_material_properties_list = np.array([False]*15,dtype=bool), use_f_tensor = False, use_C_tensor = False):
        if len(use_material_properties_list) != 15:
            raise ValueError(f"The length of use_material_properties_list must be 15, but got {len(use_material_properties_list)}.")
        super().__init__()
        self._data = load_npz_data(path)
        self._dimension = self._data[0]["positions"].shape[-1]
        self._length = len(self._data)
        self._use_material_properties_list = use_material_properties_list

        self._material_property_as_feature = np.any(use_material_properties_list)
        self._C_as_feature = use_C_tensor
        self._f_tensor_as_feature = use_f_tensor

    def __len__(self):
        """Return length of dataset.

        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.

        Args:
            idx (int): Index of training example.

        Returns:
            dict:
              trajectory = (positions, particle_type, material_property (optional), n_particles_per_example).
        """

        positions = self._data[idx]["positions"]
        positions = np.transpose(positions, (1, 0, 2))
        n_particles_per_example = positions.shape[0]
        particle_type = self._data[idx]["particle_types"]

        trajectory = {
            "positions": torch.tensor(positions).to(torch.float32).contiguous(),
            "particle_types": torch.tensor(particle_type).contiguous(),
            "n_particles_per_example": n_particles_per_example
        }
        if self._material_property_as_feature:
            material_property = self._data[idx]["material_properties"]
            material_property = material_property[self._use_material_properties_list] # filter
            material_property = np.tile(material_property, (n_particles_per_example, 1))

            trajectory["material_properties"] = torch.tensor(material_property).to(torch.float32).contiguous()
        if self._C_as_feature:
            C = self._data[idx]["C"]
            C = np.transpose(C, (1, 0, 2))
            trajectory["C"] = torch.tensor(C).to(torch.float32).contiguous()
        if self._f_tensor_as_feature:
            f_tensor = self._data[idx]["f_tensor"]
            f_tensor = np.transpose(f_tensor, (1, 0, 2))
            trajectory["f_tensor"] = torch.tensor(f_tensor).to(torch.float32).contiguous()

        return trajectory


def get_data_loader_by_samples(path, input_length_sequence, input_length_tensor_sequence, use_material_properties_list, use_f_tensor, use_C_tensor, batch_size, shuffle=True):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        input_length_tensor_sequence (int): Length of input tensor sequence
        
        [Added]
        use_material_properties_list (np.array(dtype = bool))
        use_f_tensor (bool): Whether to use f_tensor
        use_C_tensor (bool): Whether to use C_tensor

        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = SamplesDataset(path, input_length_sequence, input_length_tensor_sequence, use_material_properties_list, use_f_tensor, use_C_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)


def get_data_loader_by_trajectories(path, use_material_properties_list, use_f_tensor, use_C_tensor):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.
        use_material_properties_list  (np.array(dtype = bool))

        [Added]
        use_f_tensor (bool): Whether to use f_tensor
        use_C_tensor (bool): Whether to use C_tensor

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = TrajectoriesDataset(path, use_material_properties_list, use_f_tensor, use_C_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)