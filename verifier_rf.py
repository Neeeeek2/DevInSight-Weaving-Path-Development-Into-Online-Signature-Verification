#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import pickle
import numpy 
import torch
import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

from verifier_utils import *

numpy.set_printoptions(threshold=1e6)


'''Settings for the verifier. Note that in the preprocessing stage, the 
signatrues are sorted according to the filenames, therefore we directly 
index the samples using integers correspondting to the sorted results.
Different subsets may have slightly different indexing integers.

Accoring to the name convention, in the stylus scenario, the 6-th genuine 
signature for BSDS2 and the 7-th sample for the others are used as random 
forgeies. In the finger scenario, the 1-st genuine signature is used.'''

def get_args():
    parser = argparse.ArgumentParser(description='Online signature verification')
    parser.add_argument('--index', type=int, default=0, 
                        help='idex for training (default: 0)')
    parser.add_argument('--epoch', type=str, default="End", 
                        help='model from the i-th epoch for testing')
    parser.add_argument('--dataset', type=str, default="ChS_s1s2", 
                        help='dataset for testing')
    parser.add_argument('--way', type=str, default="fastdtw",
                        help='way for dtw calculation')
    args = parser.parse_args()
    os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0" # suppress warning of low occupancy caused by cuda-fast-dtw
    return args

if __name__ == "__main__":
    args = get_args()
    dist_weight = [0,0.5,0.5]   # weight for max, min, mean
    thre = numpy.arange(0, 20, 0.001)[None,:]

    EERs = {
        "4vs1":{
            "EER_G":0.,
            "EER_L":[]
        }, 
        "1vs1":{
            "EER_G":0.,
            "EER_L":[]
        }
    }
    datum_p = {
        "4vs1": [],
        "1vs1": [[] for _ in range(4)]
    }
    datum_n = {
        "4vs1": [],
        "1vs1": [[] for _ in range(4)]
    }
    DET_FAR = {
        "4vs1": 0., 
        "1vs1": 0.
    }
    DET_FRR = {
        "4vs1": 0.,
        "1vs1": 0.
    }

    # feats = numpy.load(f"exps/index{args.index}/{args.dataset}/feats_epoch{args.epoch}.npy")
    feats = pickle.load(open(f"exps/index{args.index}/{args.dataset}/feats_epoch{args.epoch}.pkl", "rb"))
    Dim_feats = feats[0].shape[-1]
    feats = numpy.array(feats) # only when every feature has the same shape
    labels = numpy.load(f"exps/index{args.index}/{args.dataset}/labels_epoch{args.epoch}.npy")
    N_user = len(numpy.unique(labels))
    inds = numpy.load(f"exps/index{args.index}/{args.dataset}/inds_epoch{args.epoch}.npy")

    # tricky sampling here
    if args.dataset in (
        "MCYT_stylus" \
        "BioSecurID_stylus" \
        "eBioSign1w1_stylus" "eBioSign1w2_stylus" "eBioSign1w3_stylus" "eBioSign1w4_stylus" "eBioSign1w5_stylus" \
        "eBioSign2w2_stylus"
        ):
        bias = 7
    elif args.dataset == 'BioSecureDB2_stylus':
        bias = 6
    else:
        bias = 0
    RF = []
    rf_idx = []
    for idx, l in enumerate(labels):
        if int(l) == len(RF): 
            rf_idx.append(idx+bias)
            RF.append(feats[idx+bias])
    RF = numpy.array(RF)

    pbar = tqdm(total=N_user)
    for k in numpy.arange(N_user): 
        idxUser = numpy.where(labels==k)[0]
        dset = [feats[i] for i in idxUser]; ind = inds[idxUser]

        idxGen = numpy.where(ind)[0]
        # The first 4 genuine signatures are used as templates.
        idxTemp = idxGen[:4] 
        temp = [dset[i] for i in idxTemp]
        # The rest genuine signatures and skilled forgeries are used for testing.
        idx_testG = list(set(range(len(idxGen))) - set(idxTemp))
        testG = [dset[i] for i in idx_testG]
        idx_testRF = numpy.delete(numpy.arange(len(RF)), k)
        testRF = [RF[i] for i in idx_testRF]
        test = testG + testRF
        testInd = numpy.zeros(len(test))
        testInd[:len(testG)] = 1

        ### distance calculation
        distTemp = dis_NvsN(temp, temp, args.way)   # (N_tem, N_tem)
        dist = dis_NvsN(test, temp, args.way)       # (N_test, N_tem)
        
        '''4vs1'''
        # Intra-writer statistics for score normalization. We only use the mean & min scores.
        idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
        distMax = numpy.max(dist, axis=1)[:,None] / dmax
        distMin = numpy.min(dist, axis=1)[:,None] / dmin
        distMean = numpy.mean(dist, axis=1)[:,None] / dmean
        dist41 = numpy.concatenate((distMax, distMin, distMean), axis=1)
        datum_p["4vs1"].append(dist41[numpy.where(testInd)[0]])
        datum_n["4vs1"].append(dist41[numpy.where(1-testInd)[0]])
        # local threshold
        user_p = datum_p["4vs1"][-1]
        user_n = datum_n["4vs1"][-1]
        FRR = 1. - numpy.sum(numpy.sum(user_p * dist_weight, axis=1)[:,None] - thre <= 0, axis=0) / float(user_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(user_n * dist_weight, axis=1)[:,None] - thre >= 0, axis=0) / float(user_n.shape[0])
        EER_L_this = getEER(FAR, FRR)[0] * 100
        EERs["4vs1"]["EER_L"].append(EER_L_this)

        '''1vs1'''
        # no need to consider the intra-writer statistics, as the template is the only one
        dist11 = dist[..., None].repeat(3, axis=-1)
        EER_L_this = []
        for i in range(4):  # for each template
            datum_p["1vs1"][i].append(dist11[:,i,:][numpy.where(testInd)[0]])
            datum_n["1vs1"][i].append(dist11[:,i,:][numpy.where(1-testInd)[0]])
            # local threshold
            user_p = datum_p["1vs1"][i][-1]
            user_n = datum_n["1vs1"][i][-1]
            FRR = 1. - numpy.sum(numpy.sum(user_p * dist_weight, axis=1)[:,None] - thre <= 0, axis=0) / float(user_p.shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n * dist_weight, axis=1)[:,None] - thre >= 0, axis=0) / float(user_n.shape[0])
            EER_L_this.append(getEER(FAR, FRR)[0] * 100)
        EERs["1vs1"]["EER_L"].append(numpy.mean(EER_L_this))

        pbar.update(1)

    EERs["4vs1"]["EER_L"] = numpy.mean(EERs["4vs1"]["EER_L"])
    EERs["1vs1"]["EER_L"] = numpy.mean(EERs["1vs1"]["EER_L"])

    '''4vs1'''
    datum_p["4vs1"] = numpy.concatenate(datum_p["4vs1"], axis=0)
    datum_n["4vs1"] = numpy.concatenate(datum_n["4vs1"], axis=0)
    FRR = 1. - numpy.sum(numpy.sum(datum_p["4vs1"] * dist_weight, axis=1)[:,None] - thre <= 0, axis=0) / float(datum_p["4vs1"].shape[0])
    FAR = 1. - numpy.sum(numpy.sum(datum_n["4vs1"] * dist_weight, axis=1)[:,None] - thre >= 0, axis=0) / float(datum_n["4vs1"].shape[0])
    EER_G41 = getEER(FAR, FRR)[0] * 100
    EERs["4vs1"]["EER_G"] = EER_G41
    DET_FAR["4vs1"] += FAR
    DET_FRR["4vs1"] += FRR

    '''1vs1'''
    EER_G11 = []
    for i in range(4):
        datum_p["1vs1"][i] = numpy.concatenate(datum_p["1vs1"][i], axis=0)
        datum_n["1vs1"][i] = numpy.concatenate(datum_n["1vs1"][i], axis=0)
        FRR = 1. - numpy.sum(numpy.sum(datum_p["1vs1"][i] * dist_weight, axis=1)[:,None] - thre <= 0, axis=0) / float(datum_p["1vs1"][i].shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n["1vs1"][i] * dist_weight, axis=1)[:,None] - thre >= 0, axis=0) / float(datum_n["1vs1"][i].shape[0])
        EER_G11.append(getEER(FAR, FRR)[0] * 100)
        DET_FAR["1vs1"] += FAR / 4
        DET_FRR["1vs1"] += FRR / 4
    EERs["1vs1"]["EER_G"] = numpy.mean(EER_G11)


    # save DET curve for DeepSignDB overall verification
    N_g = datum_p["4vs1"].shape[0] ; N_n = datum_n["4vs1"].shape[0]
    os.makedirs(f"DET", exist_ok=True)
    os.makedirs(f"DET/{args.index}", exist_ok=True)
    numpy.save(
        f"DET/{args.index}/{args.dataset}_rf_4vs1.npy",
        numpy.array([DET_FAR["4vs1"], DET_FRR["4vs1"], N_g, N_n], dtype=object), 
        allow_pickle=True,
    )
    numpy.save(
        f"DET/{args.index}/{args.dataset}_rf_1vs1.npy",
        numpy.array([DET_FAR["1vs1"], DET_FRR["1vs1"], N_g, N_n], dtype=object), 
        allow_pickle=True,
    )


    ### write results into json file
    import json
    import fcntl
    filename = "results.json"

    if not os.path.exists(filename):
        results = {'init': 0.0}
        json.dump(results, open(filename, "w"), indent=4)
    # read the file
    with open(filename, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX) # 锁定文件；如果文件已被锁定，那么此函数将阻塞直到文件解锁
        results = json.load(f)

        index = str(args.index)
        dataset = args.dataset
        
        def constuct(pointer, keys):
            '''keys order: [index, dataset, sfrf, Nvs1]'''
            for k in keys:
                if k not in pointer.keys():
                    pointer[k] = {}
                pointer = pointer[k]

        constuct(results, [index, dataset, "rf"])
        results[index][dataset]["rf"] = EERs

        f.seek(0) # 移动文件指针到文件开头
        f.truncate() # 清空文件内容
        json.dump(results, f, indent=4)
        f.flush() # 清空缓冲区
        fcntl.flock(f.fileno(), fcntl.LOCK_UN) # 解锁文件
  
