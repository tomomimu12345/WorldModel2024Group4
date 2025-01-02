import torch
import numpy as np


def load_npz_data(path):
    """Load data stored in npz format.

    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.

    Returns:
        data (list): List of tuples of the form (positions, particle_type).
    """
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    return data


class SamplesDataset(torch.utils.data.Dataset):
    """Dataset of samples of trajectories.
    
    Each sample is a tuple of the form (positions, particle_type, material_properties, C, f_tensor).

    positions : numpy array of shape (sequence_length, n_particles, dimension).
    particle_type : numpy array of shape (n_particles, dtype = int)
    material_properties : numpy array of shape (15, dtype = float)

    Added:
    C : numpy array of shape (sequence_length, n_particles, dimension * dimension)
    f_tensor : numpy array of shape (sequence_length, n_particles, dimension * dimension)

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input Position sequence
        input_length_tensor_sequence(int): Length of input tensor sequence
        use_material_properties_list(np.array, dtype = bool): Array of length 15 storing the flag whether to use property or not

    Attributes:
        _data (list): List of tuples of the form (positions, particle_type).
        _dimension (int): Dimension of the data.
        _input_length_sequence (int): Length of input sequence.
        _input_length_tensor_sequence (int): Length of input tensor sequence
        _data_lengths (list): List of lengths of trajectories in the dataset.
        _length (int): Total number of samples in the dataset.
        _precompute_cumlengths (np.array): Precomputed cumulative lengths of trajectories in the dataset.
    """

    def __init__(self, path, input_length_sequence = 6, input_length_tensor_sequence = 2, use_material_properties_list = np.array([False]*15, dtype=bool)):
        super().__init__()
        if input_length_sequence < input_length_tensor_sequence:
            raise ValueError("input_length_sequence must be greater than or equal to input_length_tensor_sequence")
        if len(use_material_properties_list)!=15:
            raise ValueError("the length of use_material_properties_list must be 15.")

        self._data = load_npz_data(path)
        
        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0]["positions"].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._input_length_tensor_sequence = input_length_tensor_sequence
        self._use_material_properties_list = use_material_properties_list
        
        self._material_property_as_feature = True if "material_properties" in self._data[0] else False
        self._C_as_feature = True if "C" in self._data[0] else False
        self._f_tensor_as_feature = True if "f_tensor" in self._data[0] else False

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

        if self._material_property_as_feature:
            material_property = self._data[trajectory_idx]["material_properties"]
            material_property = material_property[self._use_material_properties_list] # filter
            material_property = np.full(positions.shape[0], material_property, dtype=float)

        if self._C_as_feature:
            C = self._data[trajectory_idx]["C"][time_idx - self._input_length_tensor_sequence:time_idx]
            C = np.transpose(C, (1, 0, 2))

        if self._f_tensor_as_feature:
            f_tensor = self._data[trajectory_idx]["f_tensor"][time_idx - self._input_length_tensor_sequence:time_idx]
            f_tensor = np.transpose(f_tensor, (1, 0, 2))

        n_particles_per_example = positions.shape[0] # 粒子数
        label = self._data[trajectory_idx]["positions"][time_idx]
        label_tensor = self._data[trajectory_idx]["f_tensor"][time_idx]

        training_example = {
            "positions": positions, # (nparticles, input_sequence_length, dimension)
            "particle_types": particle_type, # (nparticles, 1)
            "n_particles_per_example": n_particles_per_example,#(1,)
            "label": label, # (nparticles, dimension)
            "label_tensor": label_tensor # (nparticles, dimension * dimension)
        }

        if self._material_property_as_feature:
            training_example["material_properties"] = material_property  # (nparticles, attributes)
        if self._C_as_feature:
            training_example["C"] = C  # (nparticles, input_sequence_length, dimension * dimension)
        if self._f_tensor_as_feature:
            training_example["f_tensor"] = f_tensor # (nparticles, input_sequence_length, dimension * dimension)

        return training_example


def collate_fn(data):
    """Collate function for SamplesDataset.
    positions : (sequence_length, n_particles, dimension)

    Args:
        data (list): List of dict {positions, particle_types, n_particles_per_example, label, ....}.

    Returns:
        dict: dict of the form {positions, particle_types, n_particles_per_example, label, ....}.   
        each array shape does not change. 
    """
    material_property_as_feature = True if "material_properties" in data[0] else False
    C_as_feature = True if "C" in data[0] else False
    f_tensor_as_feature = True if "f_tensor" in data[0] else False
    
    position_list = []
    particle_type_list = []
    if C_as_feature:
        C_list = []
    if f_tensor_as_feature:
        f_tensor_list = []
    if material_property_as_feature:
        material_property_list = []
    n_particles_per_example_list = []
    label_list = []
    label_tensor_list = []

    for comp in data:
        position_list.append(comp["positions"])
        particle_type_list.append(comp["particle_types"])
        n_particles_per_example_list.append(comp["n_particles_per_example"])
        label_list.append(comp["label"])
        label_tensor_list.append(comp["label_tensor"])

        if material_property_as_feature:
            material_property_list.append(comp["material_properties"])
        if C_as_feature:
            C_list.append(comp["C"])
        if f_tensor_as_feature:
            f_tensor_list.append(comp["f_tensor"])
    
    collated_data = {
        "positions": torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
        "particle_types": torch.tensor(np.concatenate(particle_type_list)).contiguous(),
        "n_particles_per_example": torch.tensor(n_particles_per_example_list).contiguous(),
        "label": torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous(),
        "label_tensor": torch.tensor(np.vstack(label_tensor_list)).to(torch.float32).contiguous()
    }

    if material_property_as_feature:
        collated_data["material_properties"] = torch.tensor(np.concatenate(material_property_list)).to(torch.float32).contiguous()
    if C_as_feature:
        collated_data["C"] = torch.tensor(np.vstack(C_list)).to(torch.float32).contiguous()
    if f_tensor_as_feature:
        collated_data["f_tensor"] = torch.tensor(np.vstack(f_tensor_list)).to(torch.float32).contiguous()

    return collated_data


class TrajectoriesDataset(torch.utils.data.Dataset):
    """Dataset of trajectories.

    Each trajectory is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    """

    def __init__(self, path, use_material_properties_list = np.array([False]*15, dtype=bool)):
        super().__init__()
        self._data = load_npz_data(path)
        self._dimension = self._data[0]["positions"].shape[-1]
        self._length = len(self._data)
        self._use_material_properties_list = use_material_properties_list

        self._material_property_as_feature = True if "material_properties" in self._data[0] else False
        self._C_as_feature = True if "C" in self._data[0] else False
        self._f_tensor_as_feature = True if "f_tensor" in self._data[0] else False

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
            material_property = np.full(positions.shape[0], material_properties, dtype=float)

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


def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = SamplesDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)


def get_data_loader_by_trajectories(path):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = TrajectoriesDataset(path)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)