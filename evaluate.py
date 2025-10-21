# -*- coding:utf-8 -*-
import os, pickle
import subprocess
import numpy 
import argparse
from collections import OrderedDict
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import dataset.dataset_Train_Test as dataset

from DEV_blocks_network import network


parser = argparse.ArgumentParser(description='Online signature verification')
parser.add_argument('--index', type=int, default=0, 
                    help='idex for training (default: 0)')
parser.add_argument('--seed', type=int, default=12345, 
                    help='numpy random seed (default: 12345)')
parser.add_argument('--epoch', type=str, default="End", 
                    help='model from the i-th epoch for testing')
parser.add_argument('--resampled-len', type=int, default=800, 
                    help='length to resample the signature')
parser.add_argument('--dataset', type=str, default="ChS_s1s2", 
                    help='dataset for testing')
parser.add_argument('--path', type=str, nargs='+', default=["../MSDS_process/data/data/Testdata_o.pkl"], 
                    help='path of dataset')
parser.add_argument('--device-No', type=int, default=-1, 
                    help='assign GPU device')
parser.add_argument('--m', type=int, default=8, 
                    help='Dev matrix size')
parser.add_argument('--lie', type=str, default='sp', 
                    help='Lie Algebra')
args = parser.parse_args()

dtype = torch.float32

print(f"evaluate index: {args.index}, model of epoch: {args.epoch}")
if args.device_No >= 0:
    best_gpu = args.device_No
else:
    best_gpu = args.index % 3
print(f'Selected GPU: {best_gpu}')

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)     
torch.cuda.manual_seed(args.seed)
cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True

dset = dataset.dataset(args)
for dset_path in args.path:
    print(f"loading dataset from {dset_path}")
    sigDict = pickle.load(open(dset_path, "rb"), encoding="iso-8859-1")
    dset.addDataSet(sigDict)
    del sigDict
sampler = dataset.batchSampler_test(dset)
dataLoader = DataLoader(dset, num_workers=8, batch_sampler=sampler, collate_fn=dataset.collate_fn) 

weightDict = torch.load(f"models/{args.index}/epoch{args.epoch}", map_location=torch.device('cpu'))
n_classes = weightDict.get('cls_wi.weight').shape[0] if 'cls_wi.weight' in weightDict else 1
model = network(in_dim=dset.featDim,
                n_classes=n_classes,
                n_task=1,
                n_shot_g=dset.numGen[0],
                n_shot_f=dset.numNeg[0],
                lie=args.lie,
                m=args.m,
                ).to(device=best_gpu,dtype=dtype)
model.load_state_dict(weightDict)
model.train(mode=False)
model.eval()

feats = []
# cls = []
labels = []
# conv_output = []
atten_list = []
with torch.no_grad():
    for batch in tqdm(dataLoader):
        sig, lens, mask, label = batch

        sig = torch.from_numpy(sig).to(best_gpu,dtype)
        mask = torch.from_numpy(mask).to(best_gpu,dtype)

        # o1 = sig
        # o2 = mask

        o1, o2 = model(sig, mask)

        o1 = o1.data.cpu().numpy()
        lens = o2.sum(dim=1).to(torch.int32)
        for i in range(len(lens)):
            feats.append(o1[i, :int(lens[i])])
        labels.append(label)

# feats = numpy.concatenate(feats, axis=0)
# print(feats.shape)
# cls = numpy.concatenate(cls, axis=0)
# # print(cls.shape)
labels = numpy.concatenate(labels, axis=0)
# print(label.shape)
inds = []
for i in range(len(dset.numGen)):
    inds += [
            numpy.repeat(1, dset.numGen[i]), 
            numpy.repeat(0, dset.numNeg[i])
    ]
inds = numpy.concatenate(inds, axis=0)
# print(inds.shape)

os.makedirs(f"exps/index{args.index}", exist_ok=True)

os.makedirs(f"exps/index{args.index}/{args.dataset}", exist_ok=True)

# numpy.save(f"exps/index{args.index}/{args.dataset}/feats_epoch{args.epoch}.npy", feats)
pickle.dump(feats, open(f"exps/index{args.index}/{args.dataset}/feats_epoch{args.epoch}.pkl", "wb"))
numpy.save(f"exps/index{args.index}/{args.dataset}/labels_epoch{args.epoch}.npy", labels)
numpy.save(f"exps/index{args.index}/{args.dataset}/inds_epoch{args.epoch}.npy", inds)