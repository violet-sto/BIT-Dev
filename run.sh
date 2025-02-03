#python setup.py build_ext --inplace
#python setup_cython.py build_ext --inplace
#pip install -e .
export data_path='../datasets'
export save_path="../logs/test"                        # path to logs

export n_gpu=8
export lr=2e-4                                      # peak learning rate
export warmup_steps=12000                         # warmup steps
export total_steps=200000                         # total steps
export layers=12                                    # set layers=18 for 18-layer model
export complexffn_start_layer=12                    # start layer of complex ffn
export hidden_size=768                              # dimension of hidden layers
export ffn_size=768                                 # dimension of feed-forward layers
export num_head=32                                  # number of attention heads
export batch_size=2 # 256                                 # batch size for a single gpu (ligand: 256, pocket: 8, complex: 4)
export max_positions=512                            # max tokens
export dropout=0.0
export act_dropout=0.1
export attn_dropout=0.1
export weight_decay=0.0
export sandwich_ln="false"                           # sub LayerNrom
export droppath_prob=0.1                            # probability of stochastic depth
export noise_scale=0.2                             # noise scale
export mode_prob="0.2,0.2,0.6"                      # mode distribution for {2D+3D, 2D, 3D}
#export modality="ligand,pocket,complex"             # input modality
export modality="ligand"             # input modality
export mode_ffn="false"
export mode_2d="true"
export mode_3d="true"
#export dataset_name="PCQM4M-&-PbCmQC-7M"           # 小分子数据集
#export dataset_name="PCQM4M-&-PbCmQC-79M"
#export dataset_name="PbCmQC-79M"
export dataset_name="PCQM4M-LSC-V2-3D"
export add_3d="true"
export num_3d_bias_kernel=128                       # number of Gaussian Basis kernels
export update_freq=4

export use_wandb="false"
#export WANDB_NAME="123-mode-32gpu"
#export wandb_project="BIT-kdd-ablation"
#conda activate Transformer-M

bash train.sh

