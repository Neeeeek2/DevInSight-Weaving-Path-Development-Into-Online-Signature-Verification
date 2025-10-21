#!/usr/bin/env sh
export CUDA_VISIBLE_DEVICES=0
gpu=0

index=$1
dset=$2
epoch="End"
resampled_len=800

datasets=("ChS_s1s2")   # default
if [ "$dset" = "ChS" ]; then
    datasets=("ChS_s1" "ChS_s2" "ChS_s1s2")
    # datasets=("ChS_s1s2")
elif [ "$dset" = "TDS" ]; then
    datasets=("TDS_s1" "TDS_s2" "TDS_s1s2")
    # datasets=("TDS_s1s2")
fi

for dataset in ${datasets[@]}
do
    (
        path="../MSDS_process/data/Testdata_"$dataset".pkl"
        # start=`date +%s`
        python evaluate.py \
        --index $index --epoch $epoch \
        --dataset $dataset --path $path \
        --resampled-len $resampled_len \
        --device-No $gpu 
        # echo "---------------------------------------------"
        # python classifier_with_FC.py --index $index --epoch $epoch --dataset $dataset
        # echo "---------------------------------------------"
        python verifier_sf.py --index $index --epoch $epoch --dataset $dataset
        # echo "---------------------------------------------"
        python verifier_rf.py --index $index --epoch $epoch --dataset $dataset
        # echo "---------------------------------------------"
        # end=`date +%s`
        # let time=$end-$start
        # echo "total time cost: "$time"s"
    ) &
done

wait