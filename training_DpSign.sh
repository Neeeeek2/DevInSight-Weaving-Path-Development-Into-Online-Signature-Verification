#!/usr/bin/env sh

index=$1
epochs=100      # in DpSign, epochs is 100, proven the best
resampled_len=800

dset_path=("../DeepSignDB/Processed_Data/train_DeepSignDB_stylus.pkl")
# dset_path=("../DeepSignDB/Processed_Data/train_DeepSignDB_finger.pkl")

args=(
    "--index" "$index"
    "--train-shot-g" 6
    "--train-shot-f" 6
    "--train-tasks" 4
    "--epochs" "$epochs"
    "--resampled-len" "$resampled_len"
    "--path" "$dset_path"
    "--save-interval" 50
    "--lr" 0.05     # better perform FindBestLR first
)

python main.py "${args[@]}"