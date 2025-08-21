ulimit -c unlimited

[ -z "${lr}" ] && lr=2e-4
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=1600
[ -z "${max_epoch}" ] && max_epoch=10
[ -z "${layers}" ] && layers=12
[ -z "${complexffn_start_layer}" ] && complexffn_start_layer=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=32
[ -z "${batch_size}" ] && batch_size=2
[ -z "${max_positions}" ] && max_positions=1024
[ -z "${update_freq}" ] && update_freq=2
[ -z "${seed}" ] && seed=1
[ -z "${split_idx}" ] && split_idx=0
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${data_path}" ] && data_path='/workspace/datasets'
[ -z "${save_path}" ] && save_path='../checkpoints'
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln="false"
[ -z "${droppath_prob}" ] && droppath_prob=0.0

[ -z "${loss_type}" ] && loss_type="L2"
[ -z "${std_type}" ] && std_type="std_logits"
[ -z "${readout_type}" ] && readout_type="cls"

[ -z "${save_interval}" ] && save_interval=1
[ -z "${dataset_name}" ] && dataset_name="dud-e"

[ -z "${task_idx}" ] && task_idx=0

[ -z "${add_3d}" ] && add_3d="true"
[ -z "${no_2d}" ] && no_2d="false"
[ -z "${no_save}" ] && no_save="true"
[ -z "${mode_ffn}" ] && mode_ffn="true"
[ -z "${mode_2d}" ] && mode_2d="false"
[ -z "${mode_3d}" ] && mode_3d="false"
[ -z "${use_wandb}" ] && use_wandb="false"
[ -z "${wandb_project}" ] && wandb_project="default_project"
[ -z "${no_pretrain}" ] && no_pretrain="false"
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

[ -z "${MASTER_PORT}" ] && MASTER_PORT=$RANDOM
[ -z "${WORLD_SIZE}" ] && WORLD_SIZE=1

[ -z "${save_prefix}" ] && save_prefix='exp'
[ -z "${ckpt_path}" ] && ckpt_path='../logs/checkpoint_last.pt' # set this dir


echo -e "\n\n"
echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "RANK: ${RANK}"
echo "WORLD_SIZE: ${WORLD_SIZE}"


if [[ -z "${WORLD_SIZE}" ]]
then
  ddp_options=""
else
  if (( $WORLD_SIZE == 1))
  then
	ddp_options=""
  else
    ddp_options="--nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
  fi
fi
echo "ddp_options: ${ddp_options}"
echo "==============================================================================="

hyperparams=flag-false-no_pretrain-$no_pretrain-lr-$lr-end_lr-$end_lr-tepochs-$max_epoch-wsteps-$warmup_steps-L$layers-D$hidden_size-F$ffn_size-H$num_head-SLN-$sandwich_ln-BS$((batch_size*n_gpu*WORLD_SIZE*update_freq))-CLIP$clip_norm-dp$dropout-attn_dp$attn_dropout-wd$weight_decay-dpp$droppath_prob/SEED$seed-SPLIT$split_idx-RF-$readout_type
save_dir=$save_path/$save_prefix-dataset-$dataset_name/$hyperparams
tsb_dir=$save_dir/tsb
mkdir -p $save_dir

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "batch_size: $((batch_size*n_gpu*WORLD_SIZE*update_freq))"
echo "n_layers: ${layers}"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "max_epoch: ${max_epoch}"
echo "clip_norm: ${clip_norm}"
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "sandwich_ln: ${sandwich_ln}"
echo "num_head: ${num_head}"
echo "update_freq: ${update_freq}"
echo "dropout: ${dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "act_dropout: ${act_dropout}"
echo "weight_decay: ${weight_decay}"
echo "droppath_prob: ${droppath_prob}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "data_dir: ${data_path}"
echo "ckpt_dir: ${ckpt_path}"
echo "task_idx: ${task_idx}"
echo "loss_type: ${loss_type}"
echo "std_type: ${std_type}"
echo "readout_type: ${readout_type}"
echo "save_interval: ${save_interval}"
echo "dataset_name: ${dataset_name}"
echo "==============================================================================="

# ENV
echo -e "\n\n"
echo "======================================ENV======================================"
echo 'Environment'
ulimit -c unlimited;
# echo '\n\nhostname'
# hostname
# echo '\n\nnvidia-smi'
# nvidia-smi
# echo '\n\nls -alh'
# ls -alh
# echo -e '\n\nls ~ -alh'
# ls ~ -alh
echo "torch version"
python -c "import torch; print(torch.__version__)"
echo "==============================================================================="

echo -e "\n\n"
echo "==================================ACTION ARGS==========================================="
if ( $sandwich_ln == "true")
then
  action_args="--sandwich-ln "
else
  action_args=""
fi
echo "action_args: ${action_args}"

if ( $add_3d == "true")
then
  add_3d_args="--add-3d"
else
  add_3d_args=""
fi
echo "add_3d_args: ${add_3d_args}"

if ( $no_2d == "true")
then
  no_2d_args="--no-2d"
else
  no_2d_args=""
fi
echo "no_2d_args: ${no_2d_args}"

if ( $no_save == "true" )
then
  no_save_args="--no-epoch-checkpoints"
else
  no_save_args=""
fi
echo "no_save_args: ${no_save_args}"

if ( $no_pretrain == "true" )
then
  load_pretrain_args="--remove-head"
else
  load_pretrain_args="--remove-head --finetune-from-model ${ckpt_path}"
fi
echo "load_pretrain_args: ${load_pretrain_args}"

if ( $mode_ffn == "true")
then
  mode_ffn_args="--mode-ffn"
else
  mode_ffn_args=""
fi
echo "mode_ffn_args: ${mode_ffn_args}"

if ( $mode_2d == "true")
then
  mode_2d_args="--mode-2d"
else
  mode_2d_args=""
fi
echo "mode_2d_args: ${mode_2d_args}"

if ( $mode_3d == "true")
then
  mode_3d_args="--mode-3d"
else
  mode_3d_args=""
fi
echo "mode_3d_args: ${mode_3d_args}"

if ( $use_wandb == "true")
then
  wandb_project_args="--wandb-project ${wandb_project}"
else
  wandb_project_args=""
fi
echo "wandb_project_args: ${wandb_project_args}"


echo "========================================================================================"

export NCCL_ASYNC_ERROR_HADNLING=1
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES="0"

torchrun --master_port=$MASTER_PORT $ddp_options train.py \
  --user-dir $(realpath ./Transformer-M) \
	--data-path $data_path \
	--num-workers 4 --ddp-backend=legacy_ddp \
	--dataset-name $dataset_name --valid-subset valid \
	--batch-size $batch_size --data-buffer-size 80 --seed $seed --split-idx $split_idx \
	--task graph_cl_dude --criterion graph_cl_dude --arch multiway_transformer_m_base --load-dude --mode-ffn --mode-2d --mode-3d --num-classes 1 \
	--lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
	--warmup-updates $warmup_steps --total-num-update 10000 --max-epoch $max_epoch --update-freq $update_freq \
	--encoder-layers $layers --encoder-attention-heads $num_head $add_3d_args $no_2d_args --num-3d-bias-kernel $num_3d_bias_kernel \
	--encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob \
	--attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --weight-decay $weight_decay \
	--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 $action_args --clip-norm $clip_norm $no_save_args \
	--num-decoys 64 --loss-type $loss_type --std-type $std_type --readout-type $readout_type \
	--tensorboard-logdir $tsb_dir --save-dir $save_dir $load_pretrain_args --save-interval $save_interval \
  --complexffn-start-layer $complexffn_start_layer --max-positions $max_positions \
  $mode_ffn_args $mode_2d_args $mode_3d_args $wandb_project_args
