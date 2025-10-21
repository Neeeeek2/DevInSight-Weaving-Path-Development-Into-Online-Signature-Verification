#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import pickle
import json
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
Different subsets may have slightly different indexing integers.'''

parser = argparse.ArgumentParser(description='Online signature verification')
parser.add_argument('--index', type=int, default=0, 
                    help='idex for training (default: 0)')
parser.add_argument('--epoch', type=str, default="End", 
                    help='model from the i-th epoch for testing')
parser.add_argument('--stylus', type=int, default=1, 
                    help='stylus testing or finger testing')
args = parser.parse_args()

if args.stylus == 1:
    datasets = [
        "MCYT_stylus", \
        "BioSecurID_stylus", \
        "BioSecureDB2_stylus", \
        "eBioSign1w1_stylus", "eBioSign1w2_stylus", "eBioSign1w3_stylus", "eBioSign1w4_stylus", "eBioSign1w5_stylus", \
        "eBioSign2w2_stylus"
    ]
    n_users = numpy.array([100, 132, 140, 35, 35, 35, 35, 35, 35])
else:
    datasets = [
    "eBioSign1w4_finger", "eBioSign1w5_finger", \
    "eBioSign2w5_finger", "eBioSign2w6_finger"
    ]
    n_users = numpy.array([35,35,35,35])

overall = {
    "sf": {
        "4vs1":{
            "EER_G":None,
            "EER_L":None
        },
        "1vs1":{
            "EER_G":None,
            "EER_L":None
        }
    }, 
    "rf": {
        "4vs1":{
            "EER_G":None,
            "EER_L":None
        },
        "1vs1":{
            "EER_G":None,
            "EER_L":None
        }
    }
}

for fgy in ["sf", "rf"]:
    for u_tem in ["4vs1", "1vs1"]:
        # global threshold
        DET_FAR = 0 ; DET_FRR = 0
        N_n = 0 ; N_g = 0
        for d in datasets:
            FAR, FRR, n_n, n_g = numpy.load(f"DET/{args.index}/{d}_{fgy}_{u_tem}.npy", allow_pickle=True)
            DET_FAR += FAR*n_n ; DET_FRR += FRR*n_g
            N_n += n_n ; N_g += n_g
        DET_FAR /= N_n ; DET_FRR /= N_g
        EER_G = getEER(DET_FAR, DET_FRR)[0] * 100
        overall[fgy][u_tem]["EER_G"] = EER_G
        # local threshold
        EER_L = []
        with open(f"results.json", "r") as f:
            results = json.load(f)
        for d in datasets:
            EER_L.append(results[str(args.index)][d][fgy][u_tem]["EER_L"])
        EER_L = numpy.array(EER_L)
        EER_L = numpy.sum(EER_L * n_users) / numpy.sum(n_users)
        overall[fgy][u_tem]["EER_L"] = EER_L

### write results into json file
# Try to load existing data
if os.path.exists("results.json"):
    # read the file
    with open("results.json", "r") as f:
        results = json.load(f)
else:
    results = {"init":0.0}
# update the results
name = "DeepSignDB_stylus" if args.stylus == 1 else "DeepSignDB_finger"
results[str(args.index)][name] = overall
with open("results.json", "w") as f:
    # f.write(json.dumps(results, indent=4))
    json.dump(results, f, indent=4)
    