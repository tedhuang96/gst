# GST

This is the implementation for the paper

### Learning Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory Prediction

##### [Zhe Huang](https://tedhuang96.github.io/), Ruohua Li, Kazuki Shin, [Katherine Driggs-Campbell](https://krdc.web.illinois.edu/)

published in [RA-L](https://www.ieee-ras.org/publications/ra-l/).

[[Paper](https://ieeexplore.ieee.org/abstract/document/9664278)] [[arXiv](https://arxiv.org/abs/2107.07056)] [[Project](https://sites.google.com/view/gumbel-social-transformer)]

GST is the abbreviation of our model Gumbel Social Transformer. All code was developed and tested on Ubuntu 18.04 with CUDA 10.2, Python 3.6.9, and PyTorch 1.7.1. <br/>

### Citation
If you find this repo useful, please cite
```
@article{huang2022learning,
  title={Learning Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory Prediction},
  author={Huang, Zhe and Li, Ruohua and Shin, Kazuki and Driggs-Campbell, Katherine},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  volume={7},
  number={2},
  pages={1198-1205},
  doi={10.1109/LRA.2021.3138547}
}
```

### Setup
##### 1. Create a Virtual Environment. (Optional)
```
virtualenv -p /usr/bin/python3 myenv
source myenv/bin/activate
```

##### 2. Install Packages
You can run either <br/>
```
pip install -r requirements.txt
```
or <br/>
```
pip install numpy
pip install scipy
pip install matplotlib
pip install tensorboardX
pip install torch==1.7.1
```
If you want to use `tensorboard --logdir results` to check training curves, install `tensorflow` by running
```
pip install tensorflow
```

##### 3. Create Folders and Dataset Files.
```
sh run/make_dirs.sh
sh run/create_datasets.sh
```

### Training and Evaluation on Various Configurations
To train and evaluate a model with n=1, i.e., the target pedestrian pays attention to at most one partially observed pedestrian, run
```
sh run/train_sparse.sh
sh run/eval_sparse.sh
```
To train and evaluate a model with n=1 and temporal component as a temporal convolution network, run
```
sh run/train_sparse_tcn.sh
sh run/eval_sparse_tcn.sh
```
To train and evaluate a model with full connection, i.e., the target pedestrian pays attention to all partially observed pedestrians in the scene, run
```
sh run/train_full_connection.sh
sh run/eval_full_connection.sh
```
To train and evaluate a model in which the target pedestrian pays attention to all fully observed pedestrians in the scene, run
```
sh run/train_full_connection_fully_observed.sh
sh run/eval_full_connection_fully_observed.sh
```

### Important Arguments for Building Customized Configurations
- `--spatial_num_heads_edges`: n, i.e., the upperbound number of pedestrians that the target pedestrian can pay attention to in the scene. When n=0, it is defined as full connection, i.e., the target pedestrian pays attention to all pedestrians in the scene. Default is 4.
- `--only_observe_full_period`: The target pedestrian only pays attention to fully observed pedestrians. Default is False.
- `--temporal`: Temporal component. `lstm` is Masked LSTM, and `temporal_convolution_net` is temporal convolution network. Default is `lstm`.
- `--decode_style`: Decoding style. It has to match the option `--temporal`. `recursive` matches `lstm`, and `readout` matches `temporal_convolution_net`. Default is `recursive`.
- `--ghost`: Add a ghost pedestrian in the scene to encourage sparsity. When `--spatial_num_heads_edges` is set as zero, i.e., the target pedestrian pays attention to all pedestrians in the scene, `--ghost` has to be set as False. Default is False.

### Credits
Part of the code is based on the following works and repos:

[1] Mohamed, Abduallah, et al. "Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020. [[GitHub](https://github.com/abduallahmohamed/Social-STGCNN)]

[2] Pytorch implementation of Multi-head Attention. [[Modules](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention)] [[Functional](https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L5067)]


### Contact
Please feel free to open an issue or send an email to zheh4@illinois.edu.