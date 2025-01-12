# WorldModel2024Group4
This code is designed to train a Graph Neural Simulator based on Material Point Method simulations. 

Modifications are made to the traditional GNS to simultaneously predict the acceleration and the velocity of the deformation gradient.

![rollout_ex0](https://github.com/user-attachments/assets/2431bf53-ed86-47e2-bea6-ec0724bdb1fd)



## Environment
```
Python 3.10.12
CUDA 12.2
```

## Installation
```
git clone --recursive https://github.com/tomomimu12345/WorldModel2024Group4.git
```
```
cd WorldModel2024Group4/
```
```
pip install -r requirements.txt
```

- Install optional dependencies of pytorch_geometric for cuda and pytorch versions([pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

## Generate mpm train data
### Output in h5 file
```
python3 gen_data.py
```
Data is stored in sim_results/mpm-*.
### Conversion to npz file
```
python3 convert_hdf5_to_npz_with_Tensor.py --path $(cat train_paths.txt) --output train
python3 convert_hdf5_to_npz_with_Tensor.py --path $(cat valid_paths.txt) --output valid
python3 convert_hdf5_to_npz_with_Tensor.py --path $(cat test_paths.txt) --output test
```
### Move npz and json file to data/
```
mkdir data/
mv *.npz data/
mv train.json data/metadata.json
```

### Train
Requires at least 18 GB of VRAM for Training
```
python3 train.py --mode train --batch_size 2 --data_path data/ --validation_interval 1000 --ntraining_steps 1000000 --nsave_steps 5000 
```

### rollout simulation
#### simulation
```
python3 train.py --mode rollout --data_path data/ --model_file model-1000000.pt
```
#### render simulation
```
python3 gns_with_tensor/render_rollout.py --rollout_dir rollouts/ --rollout_name rollout_ex0 --step_stride 5
```

## PhysGaussian Demo
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
I had an error installing simple_knn, so I made the following changes to "gaussian-splatting/submodules/simple_knn/simple_knn.cu"
```
// addition
#include <float.h>
```

### run MPM simulation
```
python gs_simulationMPM.py --model_path ./gs_model/model/collapse-trained/ --output_path ./gs_model/output_video --config ./gs_model/config/collapse_config.json --render_img --compile_video --white_bg --output_h5
```
### generate Gaussian Splatting trajectry 
```
python3 convert_hdf5_to_npz_with_Tensor.py --path gs_model/output_video/simulation_h5 --output gaussian
```

## Acknowledgements
- [Graph Network Simulator](https://github.com/geoelements/gns)
- [Material Point Method Simulator](https://github.com/zeshunzong/warp-mpm)
- [Kolmogorov-Arnold Network](https://github.com/Blealtan/efficient-kan)
- [PhysGaussian](https://github.com/XPandora/PhysGaussian)