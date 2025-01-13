# WorldModel2024Group4
This code is designed to train a Graph Neural Simulator based on Material Point Method simulations. 

Modifications are made to the traditional GNS to simultaneously predict the acceleration and the velocity of the deformation gradient.

## Simulation Result: MPM vs GNS

|MPM|GNS|
|:-:|:-:|
|<video src="https://github.com/user-attachments/assets/1dd10bd6-520b-427e-a430-3d5a51ed5f03">|<video src="https://github.com/user-attachments/assets/00b79b21-0fda-4b16-8599-7d7282f7b945">|

![rollout_ex0](https://github.com/user-attachments/assets/2431bf53-ed86-47e2-bea6-ec0724bdb1fd)

## Environment
```
Python 3.10.12
CUDA 12.2
```

## Installation
### 1. Clone the repository with submodules:
```
git clone --recursive https://github.com/tomomimu12345/WorldModel2024Group4.git & cd WorldModel2024Group4/
```
### 2. Install required dependencies:
```
pip install -r requirements.txt
```
### 3. Install optional dependencies for PyTorch Geometric (adjust versions for your CUDA/PyTorch):
- [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

## Generate mpm train data
### 1. Run MPM Simulation
- Output in h5 file
```
python3 gen_data.py
```
Data will be stored in the sim_results/ directory (e.g., sim_results/mpm-*).
### 2. Convert HDF5 to NPZ
Convert simulation results to NPZ format:
```
python3 convert_hdf5_to_npz_with_Tensor.py --path $(cat train_paths.txt) --output train
python3 convert_hdf5_to_npz_with_Tensor.py --path $(cat valid_paths.txt) --output valid
python3 convert_hdf5_to_npz_with_Tensor.py --path $(cat test_paths.txt) --output test
```
### 3. Organize Data 
- Move npz and json file to data/
```
mkdir data/
mv *.npz data/
mv train.json data/metadata.json
```

## Train
Training the GNS model requires **at least 18 GB of VRAM**. Run the training script as follows:

```
python3 train.py --mode train --batch_size 2 --data_path data/ --validation_interval 1000 --ntraining_steps 1000000 --nsave_steps 5000 
```

## rollout simulation
### 1. Simulate Rollouts
Use the trained model to simulate rollouts:
```
python3 train.py --mode rollout --data_path data/ --model_file model-1000000.pt
```
### 2. Render Simulations
Render simulation results into images or videos:
```
python3 gns_with_tensor/render_rollout.py --rollout_dir rollouts/ --rollout_name rollout_ex0 --step_stride 5
```

## PhysGaussian Demo
### 1. Install Dependencies
```
cd PhysGaussian
```
```
pip install opencv_python opencv_python_headless Pillow plyfile PyMCubes==0.1.6 scipy setuptools taichi==1.5.0
```
```
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
```
```
pip install -e gaussian-splatting/submodules/simple-knn/
cd ..
```
If you encounter an error installing simple_knn, modify simple_knn.cu as follows:
```
// addition
#include <float.h>
```

### 2. run MPM simulation
Run the MPM simulation using PhysGaussian:
```
python gs_simulationMPM.py --model_path ./gs_model/model/collapse-trained/ --output_path ./gs_model/output_video --config ./gs_model/config/collapse_config.json --render_img --compile_video --white_bg --output_h5
```
### 3. generate Gaussian Splatting trajectry 
```
python3 convert_hdf5_to_npz_with_Tensor.py --path gs_model/output_video/simulation_h5 --output gaussian
```
```
mkdir gs_model/npz_data
mv gaussian.npz gs_model/npz_data/
mv gaussian.json gs_model/npz_data/metadata.json
```
### 4. Run Gaussian Splatting GNS Simulation
Simulate with Gaussian Splatting using **at least 40 GB of VRAM**
```
python3 gs_simulationGNS.py --model_path ./gs_model/model/collapse-trained/ --output_path ./gs_model/output_video_GNS --config ./gs_model/config/collapse_config.json --render_img --compile_video --white_bg --output_h5 --data_path ./gs_model/npz_data/ --model_file models/model-1000000.pt
```



## Acknowledgements
- [Graph Network Simulator](https://github.com/geoelements/gns)
- [Material Point Method Simulator](https://github.com/zeshunzong/warp-mpm)
- [Kolmogorov-Arnold Network](https://github.com/Blealtan/efficient-kan)
- [PhysGaussian](https://github.com/XPandora/PhysGaussian)
