# A Generalist Cross-Domain Molecular Learning Framework for Structure-Based Drug Discovery

This repository is the official implementation of “A Generalist Cross-Domain Molecular Learning Framework for Structure-Based Drug Discovery”, based on the official implementation of [Transformer-M](https://github.com/lsj2408/Transformer-M) and [Fairseq](https://github.com/facebookresearch/fairseq) in [PyTorch](https://github.com/pytorch/pytorch).

## Installation

- Clone this repository

```shell
git clone https://github.com/violet-sto/BIT-Dev.git
```

- Install the dependencies (Using [Anaconda](https://www.anaconda.com/), tested with CUDA version 11.0)

```shell
cd ./BIT-Dev
conda env create -f requirement.yaml
conda activate BIT
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_geometric==1.6.3
pip install torch_scatter==2.0.7
pip install torch_sparse==0.6.9
pip install azureml-defaults
pip install rdkit-pypi cython
python setup.py build_ext --inplace
python setup_cython.py build_ext --inplace
pip install -e .
pip install --upgrade protobuf==3.20.1
pip install --upgrade tensorboard==2.9.1
pip install --upgrade tensorboardX==2.5.1
```

## Training

```shell
export lr=2e-4                                      # peak learning rate
export warmup_steps=12000                           # warmup steps
export total_steps=200000                           # total steps
export layers=12                                    # set layers=18 for 18-layer model
export hidden_size=768                              # dimension of hidden layers
export ffn_size=768                                 # dimension of feed-forward layers
export num_head=32                                  # number of attention heads
export batch_size=512                               # batch size for a single gpu
export dropout=0.0
export act_dropout=0.1
export attn_dropout=0.1
export weight_decay=0.0
export droppath_prob=0.1                            # probability of stochastic depth
export noise_scale=0.2                              # noise scale
export mode_prob="0.2,0.2,0.6"                      # mode distribution for {2D+3D, 2D, 3D}
export modality="ligand,pocket,complex"
export add_3d="true"
export num_3d_bias_kernel=128                       # number of Gaussian Basis kernels
bash train.sh
```

Our model is trained on 64 NVIDIA Tesla V100 GPUs (32GB).
