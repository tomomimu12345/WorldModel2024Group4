import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from gns_with_tensor.data_loader import SamplesDataset, collate_fn

class TestSamplesDatasetWithRealData(unittest.TestCase):

    def setUp(self):
        # Path to the real dataset
        self.mock_path = "sample.npz"

        # Test configurations
        self.input_length_sequence = 6
        self.input_length_tensor_sequence = 2
        self.use_material_properties_list = np.array([False] * 15, dtype=bool)
        self.use_f_tensor = False
        self.use_C_tensor = False
        self.batch_size = 16

    def test_dataset_initialization(self):
        dataset = SamplesDataset(
            path=self.mock_path,
            input_length_sequence=self.input_length_sequence,
            input_length_tensor_sequence=self.input_length_tensor_sequence,
            use_material_properties_list=self.use_material_properties_list,
            use_f_tensor=self.use_f_tensor,
            use_C_tensor=self.use_C_tensor
        )
        print(f"Dataset length: {len(dataset)}")
        print(f"Dimension: {dataset._dimension}")
        self.assertGreater(len(dataset), 0, "Dataset should not be empty.")
        self.assertEqual(dataset._dimension, 3)

    def test_dataset_getitem(self):
        dataset = SamplesDataset(
            path=self.mock_path,
            input_length_sequence=self.input_length_sequence,
            input_length_tensor_sequence=self.input_length_tensor_sequence,
            use_material_properties_list=self.use_material_properties_list,
            use_f_tensor=self.use_f_tensor,
            use_C_tensor=self.use_C_tensor
        )

        sample = dataset[0]
        print("Sample keys:", sample.keys())
        self.assertEqual(sample["positions"].shape[1:], (self.input_length_sequence, 3))
        self.assertEqual(sample["particle_types"].shape[0], sample["positions"].shape[0])
        if self.use_material_properties_list.any():
            self.assertIn("material_properties", sample)
        if self.use_C_tensor:
            self.assertIn("C", sample)
        if self.use_f_tensor:
            self.assertIn("f_tensor", sample)

    def test_collate_fn(self):
        dataset = SamplesDataset(
            path=self.mock_path,
            input_length_sequence=self.input_length_sequence,
            input_length_tensor_sequence=self.input_length_tensor_sequence,
            use_material_properties_list=self.use_material_properties_list,
            use_f_tensor=self.use_f_tensor,
            use_C_tensor=self.use_C_tensor
        )
        data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        batch = next(iter(data_loader))
        print(f"Batch positions shape: {batch['positions'].shape}")
        print(f"Batch particle_types shape: {batch['particle_types'].shape}")
        self.assertGreater(len(batch["positions"]), 0, "Batch should contain positions.")
        self.assertGreater(len(batch["particle_types"]), 0, "Batch should contain particle types.")
        if "material_properties" in batch:
            print("Material properties present in batch.")
        if "C" in batch:
            print("C tensor present in batch.")
        if "f_tensor" in batch:
            print("f_tensor present in batch.")

if __name__ == "__main__":
    unittest.main()
