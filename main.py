# -*- coding:utf-8 -*-
import os, pickle
import numpy 
import argparse
from collections import OrderedDict

import time
from tqdm import tqdm
import logging
import pdb
import sys
import subprocess
# from thop import profile
import torch
import torch.distributed as dist
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
parser.add_argument('--train-shot-g', type=int, default=6, 
                    help='number of genuine samples per class per training batch(default: 5)')
parser.add_argument('--train-shot-f', type=int, default=6, 
                    help='number of forgery samples per class per training batch(default: 5)')
parser.add_argument('--train-tasks', type=int, default=4, 
                    help='number of tasks per batch')
parser.add_argument('--epoch_start', type=int, default=0,
                    help='start epoch')
parser.add_argument('--epochs', type=int, default=200, 
                    help='number of epochs to train (default: 200)')
parser.add_argument('--resampled-len', type=int, default=800, 
                    help='length to resample the signature')
parser.add_argument('--path', type=str, nargs='+',
                    default=["../MSDS_process/data/Traindata_ChS_s1s2.pkl"],
                    help='path of dataset')
parser.add_argument('--seed', type=int, default=12345, 
                    help='numpy random seed')
parser.add_argument('--save-interval', type=int, default=50, 
                    help='how many epochs to wait before saving the model.')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='learning rate')
parser.add_argument('--device-No', type=int, default=-1, 
                    help='assign GPU device')
parser.add_argument('--m', type=int, default=8, 
                    help='Dev matrix size')
parser.add_argument('--lie', type=str, default='sp', 
                    help='Lie Algebra')
args = parser.parse_args()


os.makedirs(f"models/{args.index}", exist_ok=True)
logging.basicConfig(
        filename=f"models/{args.index}/train.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

n_task = args.train_tasks
n_shot_g = args.train_shot_g
n_shot_f = args.train_shot_f
dtype = torch.float32

print(f'training index: {args.index}')
if args.device_No >= 0:
    best_gpu = args.device_No
else:
    best_gpu = args.index % 3
print(f'Selected GPU: {best_gpu}')
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0" # suppress warning of low occupancy caused by cuda-fast-dtw

dset = dataset.dataset(args)
for dset_path in args.path:
    print(f"loading dataset from {dset_path}")
    sigDict = pickle.load(open(dset_path, "rb"), encoding="iso-8859-1")
    dset.addDataSet(sigDict)
    del sigDict
sampler = dataset.batchSampler_train(dset, loop=False, taskSize=n_task, taskNumGen=n_shot_g, taskNumNeg=n_shot_f)
dataLoader = DataLoader(dset, num_workers=4, batch_sampler=sampler, collate_fn=dataset.collate_fn) 

model = network(in_dim=dset.featDim,
                n_classes=len(dset), 
                n_task=n_task,
                n_shot_g=n_shot_g,
                n_shot_f=n_shot_f,
                lie = args.lie,
                m = args.m,
                ).to(device=0,dtype=dtype)
model.train(mode=True)
if args.epoch_start > 0:
    model.load_state_dict(torch.load(f"models/{args.index}/epoch{args.epoch_start}"))
    print(f"loading model from models/{args.index}/epoch{args.epoch_start}")

# n_parameters = sum(p.numel() for p in model.parameters())
# print('Number of params:', n_parameters)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
# optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=10)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

def adjust_learning_rate(optim, epoch):
    """Sets the learning rate schedule"""
    warm_up = args.epochs // 20
    if epoch < warm_up:
        lr = args.lr * (epoch + 1) / warm_up
    else:
        # # no schedule
        # lr = args.lr
        # # original setting
        # lr = args.lr if epoch<100 else args.lr*0.1
        # # exp decay
        # lr = args.lr * 0.98**(epoch)
        # cos decay
        lr = args.lr * 0.5 * (1 + numpy.cos((epoch - warm_up) * numpy.pi / (args.epochs - warm_up)))
        # # lr lower bound
        # lr = lr if lr > 1e-1*args.lr else 1e-1*args.lr
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def predict_acc(logits, label):
    one_hot = torch.zeros_like(logits).scatter(1, label.view(-1, 1).to(torch.int64), 1)
    one_hot = one_hot.data.cpu().numpy()
    logits = logits.data.cpu().numpy()
    pre = numpy.zeros(logits.shape)
    pre[numpy.arange(len(logits)), numpy.argmax(logits,axis=1)] = 1
    acc = numpy.sum(pre * one_hot)/(len(logits))
    return acc

### log everything
logging.info(f"Args: {args}")
logging.info(f"Model: {model}")
logging.info(f"Optimizer: {optimizer}")
# logging.info(f"LR Scheduler: {vars(lr_scheduler)}")


pbar = tqdm(total=(args.epochs-args.epoch_start)*len(dataLoader))
log_losses = {
    "loss3": 0.0, 
    "g~g": 0.0,
    "g~sf": 0.0,
    "g~rf": 0.0,
}
for epoch in range(args.epoch_start, args.epochs):
    pbar.set_description(f"Epoch {epoch+1}/{args.epochs}")
    StepsPerEpoch = len(dataLoader)
    adjust_learning_rate(optimizer, epoch)
    for batch in dataLoader:
        sig, lens, mask, label = batch
        sig = torch.from_numpy(sig).to('cuda',dtype)
        mask = torch.from_numpy(mask).to('cuda',dtype)
        # label = torch.from_numpy(label).to('cuda',dtype=torch.int64)

        model.zero_grad()
        y, mask = model(sig, mask)

        # lens = torch.from_numpy(lens * y.shape[1] / sig.shape[1]).to(0,torch.int32)
        lens = mask.sum(dim=1).to(torch.int32)
        loss_tri, \
            intra_g, intra_f, inter_fa = model.measurement_Loss(y, lens, margin=1.)
        loss = loss_tri + 0.01*intra_g
        loss.backward()

        # gradient cliping
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        
        optimizer.step()

        log_losses["loss3"] += loss_tri.item() / StepsPerEpoch
        log_losses["g~g"] += intra_g.item() / StepsPerEpoch
        log_losses["g~sf"] += intra_f.item() / StepsPerEpoch
        log_losses["g~rf"] += inter_fa.item() / StepsPerEpoch

        pbar.update(1)
        pbar.set_postfix(
            {
                "loss3": loss_tri.item(),
                "g~g": intra_g.item(),
                "g~sf": intra_f.item(),
                "g~rf": inter_fa.item(),
                "lr": optimizer.param_groups[0]['lr']
                }
            )

    ### log the information
    log_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] Epoch {epoch+1} => "
    for key, value in log_losses.items():
        log_message += f"{key}: {value:.6f} "
    log_message += f"lr: {optimizer.param_groups[0]['lr']:.2e} "
    logging.info(log_message)
    log_losses = {key: 0. for key in log_losses.keys()}   # reset the log_losses

    ### save the model
    if epoch % args.save_interval == 0:
       torch.save(model.state_dict(), f"models/{args.index}/epoch{epoch}")
    
    # # release cache for each epoch
    # torch.cuda.empty_cache()
    # scheduler.step()

torch.save(model.state_dict(), f"models/{args.index}/epochEnd")
