#!/usr/bin/env sh

index=$1
dset=$2
epochs=200
resampled_len=800
seed=$3


dset_path="../MSDS_process/data/Traindata_ChS_s1s2.pkl"    # default
if [ "$dset" = "ChS" ]; then
    dset_path="../MSDS_process/data/Traindata_ChS_s1s2.pkl"
elif [ "$dset" = "TDS" ]; then
    dset_path="../MSDS_process/data/Traindata_TDS_s1s2.pkl"
fi


args=(
    "--index" "$index"
    "--train-shot-g" 6
    "--train-shot-f" 6
    "--train-tasks" 4
    "--epoch_start" 0
    "--epochs" "$epochs"
    "--resampled-len" "$resampled_len"
    "--path" "$dset_path"
    "--save-interval" 50
    "--lr" 0.001    # in MSDS, lr is 1e-3, proven the best
    "--seed" $seed
)

python main.py "${args[@]}"