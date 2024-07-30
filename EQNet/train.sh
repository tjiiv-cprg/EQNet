#!/usr/bin/env bash -l

ulimit -SHn 51200
ulimit -s unlimited

# eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RAMoPred


# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_MEMEFF_ATT=False
export MAX_EPOCHS=16
export T_MAX=16
export INITIAL_LR=5e-4
export GRAD_ACC=1

if [[ $1 == "-c" ]]; then # CPU debug mode
    export USE_MEMEFF_ATT=False
    export CUDA_VISIBLE_DEVICES=
    NUM_DEVICES=1
    BATCH_SIZE=2
    NUM_WORKERS=0
elif [[ $1 == "-d" ]]; then # GPU debug mode
    export CUDA_LAUNCH_BLOCKING=1
    NUM_DEVICES=1
    BATCH_SIZE=8
    NUM_WORKERS=0
elif [[ $1 == "-m" ]]; then # manual mode
    export PYTHONOPTIMIZE=1
    NUM_DEVICES=2
    BATCH_SIZE=16
    NUM_WORKERS=8
    T_MAX=64
    MAX_EPOCHS=64
elif [ -z $1 ]; then
    echo "execution mode is mandatory."
    exit 1
else
    export PYTHONOPTIMIZE=1
    RECIPE=$1
fi

python EQNet/train_qcnet_shq.py \
    --train_batch_size $BATCH_SIZE --val_batch_size $BATCH_SIZE --test_batch_size $BATCH_SIZE --devices $NUM_DEVICES \
    --max_epochs $MAX_EPOCHS --T_max $T_MAX --lr 5e-4 --num_workers $NUM_WORKERS \
    --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 \
    --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 --grad_acc $GRAD_ACC \
    --brier_loss --refine --machine 3090-C
