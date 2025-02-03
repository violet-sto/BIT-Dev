export data_path='/mnt/pw-bio-bigmodel/bio-datasets/v0.1/datasets/'               # path to data
export save_path='/mnt/pw-bio-bigmodel/runlogs/v0.1/'                  # path to logs
#export data_path='../datasets'                                   # path to data
#export save_path='../logs/'                                      # path to logs

export lr=2e-4                                      # peak learning rate
export warmup_steps=150000                          # warmup steps
export total_steps=3000000                          # total steps
export layers=12                                    # set layers=18 for 18-layer model
export complexffn_start_layer=10                    # start layer of complex ffn
export hidden_size=768                              # dimension of hidden layers
export ffn_size=768                                 # dimension of feed-forward layers
export num_head=32                                  # number of attention heads
export batch_size=2                                 # batch size for a single gpu
export max_positions=1024                            # max tokens
export dropout=0.0
export act_dropout=0.1
export attn_dropout=0.1
export weight_decay=0.0
export sandwich_ln="true"                           # sub LayerNrom
export droppath_prob=0.1                            # probability of stochastic depth
export noise_scale=0.2                              # noise scale
export mode_prob="0.2,0.2,0.6"                      # mode distribution for {2D+3D, 2D, 3D}
export modality="ligand,pocket,complex"                           # input modality
export dataset_name="PCQM4M-LSC-V2-3D"
export add_3d="true"
export num_3d_bias_kernel=128                       # number of Gaussian Basis kernels
bash train_cluster.sh
