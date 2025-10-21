#!/usr/bin/env sh
export CUDA_VISIBLE_DEVICES=0
gpu=0

index=$1
epoch="End"
# epoch=100
resampled_len=800

datasets=(
    "MCYT_stylus" \
    "BioSecurID_stylus" \
    "BioSecureDB2_stylus" \
    "eBioSign1w1_stylus" "eBioSign1w2_stylus" "eBioSign1w3_stylus" "eBioSign1w4_stylus" "eBioSign1w5_stylus" \
    "eBioSign2w2_stylus" \
    "eBioSign1w4_finger" "eBioSign1w5_finger" \
    "eBioSign2w5_finger" "eBioSign2w6_finger"
    )

for dataset in ${datasets[@]}
do 
    # echo $dataset
    path="../DeepSignDB/Processed_Data/test_"$dataset".pkl"
    # start=`date +%s`
    python evaluate.py \
    --index $index --epoch $epoch \
    --dataset $dataset --path $path \
    --resampled-len $resampled_len \
    --device-No $gpu 
    # echo "---------------------------------------------"
    python verifier_sf.py --index $index --epoch $epoch --dataset $dataset &
    # echo "---------------------------------------------"
    python verifier_rf.py --index $index --epoch $epoch --dataset $dataset &
    wait
    echo "---------------------------------------------"
done

python verifier_sfrf_DeepSignDB.py --index $index --epoch $epoch --stylus 1
python verifier_sfrf_DeepSignDB.py --index $index --epoch $epoch --stylus 0