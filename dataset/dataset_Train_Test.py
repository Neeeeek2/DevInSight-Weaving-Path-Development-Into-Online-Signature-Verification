# -*- coding:utf-8 -*-
import numpy, sys
import pickle 
from matplotlib import pyplot as plt
from scipy import interpolate
import torch
import torch.nn.functional as F

from .utils import *

class dataset(object):
    """docstring for dataset"""
    def __init__(self, 
                 args
                 ):
        super(dataset, self).__init__()
        self.Keys = None
        self.numGen = None
        self.numNeg = None
        self.feats = []
        self.resampled_len = args.resampled_len

    def addDataSet(self, sigDict):
        testKeys_add = list(sigDict.keys())
        numGen_add = numpy.zeros(len(testKeys_add), dtype=numpy.int32)
        numNeg_add = numpy.zeros(len(testKeys_add), dtype=numpy.int32)

        for idx, key in enumerate(testKeys_add):
            sys.stdout.write(">>>>> User key: %s <<<<<\r"%key)
            # genunine
            for tool in sigDict[key][True].keys():
                featExt(sigDict[key][True][tool], self.feats, tool)
                numGen_add[idx] += len(sigDict[key][True][tool])
            # forgery
            for tool in sigDict[key][False].keys():
                featExt(sigDict[key][False][tool], self.feats, tool)
                numNeg_add[idx] += len(sigDict[key][False][tool])
        if self.Keys is None:
            self.Keys = testKeys_add
            self.numGen = numGen_add
            self.numNeg = numNeg_add
        else:
            self.Keys += testKeys_add
            self.numGen = numpy.concatenate((self.numGen, numGen_add), axis=0)
            self.numNeg = numpy.concatenate((self.numNeg, numNeg_add), axis=0)
        self.accumNum2 = numpy.cumsum(self.numGen + self.numNeg)
        self.accumNum = numpy.roll(self.accumNum2, 1); self.accumNum[0] = 0

        self.featDim = self.feats[0].shape[1]
        self.lens = numpy.zeros(len(self.feats), dtype=numpy.int32)
        for i, f in enumerate(self.feats):
            self.lens[i] = f.shape[0]

    def __getitem__(self, index):
        sig = self.feats[index]
        while sig.shape[0] > 2000:
            sig = self.resample(sig, sig.shape[0]//2)
        # sig = self.randomcrop(sig, (0.8,1.))
        sig = self.resample(sig, self.resampled_len)
        # sig = self.resample_by_seg(sig, 800)
        # sig = self.add_GaussianNoise(sig, noise_level=0.1, prob=0.5)
        sigLen = sig.shape[0]
        sigLabel = numpy.sum(index>=self.accumNum2) # Note that keys start from 1, while labels start from 0. 

        return sig, sigLen, sigLabel

    def __len__(self):
        return len(self.Keys) 

    def randomcrop(self, sig, p=(0.6,1.0)):
        '''
        randomly crop
        ---
        `sig`: ndarry of shape (T,C) \\
        `p`: proportion range of the orginal length for cropping, tuple \\
        `tar_len`: target length to resample to
        '''
        T,C = sig.shape
        p = numpy.random.rand(1)*(p[1]-p[0]) + p[0]
        # cropped length should be no greater than T, but no less than 100 (approximately 1 character)
        len_crop = numpy.minimum(numpy.maximum(int(p*T), 100), T)
        bias = numpy.random.randint(T-len_crop+1)
        sig_cropped = sig[bias:bias+len_crop]
        return sig_cropped

    def resample(self, sig, tar_len=500):
        '''
        resample
        ---
        sig: ndarry of shape (T,C)
        tar_len: target length to resize to
        '''
        sig = torch.from_numpy(sig).transpose(-1,-2)
        sig = F.interpolate(sig[None,:], size=tar_len, mode='linear', align_corners=False) # (1,C,tar_len)
        return sig.transpose(-1,-2).numpy().squeeze() # (tar_len,C)

    def resample_by_seg(self, sig, tar_len=500):
        '''
        resample by segmentation
        ---
        `sig`: ndarry of shape (T,C) \\
        `tar_len`: target length to resize to
        '''
        strokes = self.cur_seg(sig)
        len_new = 0
        for i, stroke in enumerate(strokes):
            if i is len(strokes)-1:
                len_new_stroke = tar_len - len_new
            else:
                len_new_stroke = int(tar_len * len(stroke) / len(sig))
            if len_new_stroke > 1:
                # stroke may be too short to resample, we skip it
                strokes[i] = self.resample(stroke, len_new_stroke)
            len_new += len(strokes[i])
        sig_new = numpy.concatenate(strokes, axis=0)
        return sig_new
    
    def cur_seg(self, sig, k=1, beta_thre=0.2):
        '''
        segment signature by curvature
        ---
        `sig`: numpy array of shape (L, C), the signature data
        `k`: int, parameter of context for every points
        `beta_thre`: float, the threshold of curvature
        `return`: list of numpy array, the segmented signature data
        '''
        x = sig[:,0] ; y = sig[:,1]
        cur_x = numpy.abs(x[2*k:]+x[:-2*k]-2*x[k:-k])
        cur_y = numpy.abs(y[2*k:]+y[:-2*k]-2*y[k:-k])
        cur = numpy.concatenate([cur_x[:,None], cur_y[:,None]], axis=-1)
        beta = numpy.max(cur, axis=-1) / 2 / k
        beta = numpy.concatenate([numpy.zeros(k), beta, numpy.zeros(k)], axis=0)
        beta /= numpy.max(beta)

        # # draw sig with beta hotmap
        # plt.figure()
        # plt.scatter(sig[:,0], -sig[:,1], c=beta, cmap='hot', alpha=0.5, s=5)
        # plt.title('Curvature')
        # plt.colorbar()
        # plt.savefig('curvature.png')
        # plt.close()

        # segment by beta
        idx_thre = numpy.where(beta>beta_thre)[0]
        if len(idx_thre) == 0:
            return [sig]
        idx_thre = numpy.concatenate([[0], idx_thre, [len(sig)]])
        # A stroke may report 2 idx_thre in start, and may also report 2 idx_thre in end
        # we must check to insure the correctness of segmentation
        diff_i = numpy.diff(idx_thre)
        diff_is = numpy.concatenate([[10],diff_i])
        starts = numpy.where(diff_is>1)[0]
        diff_ie = numpy.concatenate([diff_i,[10]])
        ends = numpy.where(diff_ie>1)[0]
        idx_seg = []
        for s, e in zip(starts, ends):
            if s==e: idx_seg.append(idx_thre[s])
            elif s<e: 
                idx_seg.append(idx_thre[(s+e)//2+1])
            else: raise ValueError('Segment Error: s > e')
        # segmenting
        strokes = []
        for i in range(len(idx_seg)-1):
            stroke = sig[idx_seg[i]:idx_seg[i+1]]
            strokes += [stroke]

        return strokes

    def add_GaussianNoise(self, sig, noise_level=0.1, prob=0.5):
        '''
        add Gaussian noise
        ---
        sig: ndarry of shape (T,C)
        noise_level: noise level
        prob: probability of adding noise
        '''
        noise = numpy.random.normal(0, noise_level, sig.shape)
        mask = numpy.random.uniform(0, 1, sig.shape) < prob
        sig = sig + noise*mask
        return sig


class batchSampler_train(object):
    """docstring for sampler"""
    def __init__(self, dataset: dataset, loop=False, taskSize=4, taskNumGen=6, taskNumNeg=6):
        super().__init__()
        self.taskSize = taskSize
        self.taskNumGen = taskNumGen
        self.taskNumNeg = taskNumNeg
        self.index = numpy.arange(0, len(dataset.Keys), dtype=numpy.int32)
        # self.index = numpy.repeat(self.index, self.taskSize, axis=0)
        self.numGen = dataset.numGen
        self.numNeg = dataset.numNeg
        self.accumNum = dataset.accumNum
        self.numIters = len(dataset)
        self.loop = loop

    def __iter__(self):
        batch = []
        numpy.random.shuffle(self.index)
        for i in range(self.numIters):
            if self.loop:
                idxs = self.index[numpy.arange(i, i+self.taskSize)%len(self.index)]
            else:
                idxs = numpy.random.choice(self.index, size=self.taskSize, replace=False)
            for idx in idxs:
                gen = numpy.random.choice(self.numGen[idx], size=self.taskNumGen, replace=False) 
                ## SF
                neg = numpy.random.choice(self.numNeg[idx], size=self.taskNumNeg, replace=False) + self.numGen[idx]
                batch.append(gen + self.accumNum[idx])
                # batch.append(gen + self.accumNum[idx])
                batch.append(neg + self.accumNum[idx])
                # batch.append(neg + self.accumNum[idx])
                # # SF + RF
                # neg = numpy.random.choice(self.numNeg[idx], size=self.taskNumNeg//2, replace=False) + self.numGen[idx]
                # batch.append(gen + self.accumNum[idx])
                # batch.append(neg + self.accumNum[idx])
                # idxs_RF = (idx+numpy.random.randint(1, len(self.index), size=self.taskNumNeg//2))%len(self.index)
                # for idx in idxs_RF:
                #     neg = numpy.random.choice(self.numGen[idx], size=1, replace=False) 
                #     batch.append(neg + self.accumNum[idx])
            batch = numpy.concatenate(batch, axis=0).astype(numpy.int32)
            yield batch
            batch = []

    def __len__(self):
        return self.numIters


def getMask(lens):
    N = len(lens); D = numpy.max(lens)
    mask = numpy.zeros((N, D), dtype=numpy.float32)
    for i in range(N):
        mask[i, :lens[i]] = 1.0
    return mask

def collate_fn(batch):
    '''
    `batch` is a list of tuple where 
    1-st element is the signature, 
    2-nd element is the len.
    3-rd element is the label.
    '''
    batchSize = len(batch)
    sig = [item[0] for item in batch]
    sigLen = numpy.array([item[1] for item in batch], dtype=numpy.int32)
    sigmask = getMask(sigLen)
    sigLabel = numpy.array([item[2] for item in batch], dtype=numpy.int32)

    # padding
    maxLen = numpy.max(sigLen)
    sigPadded = numpy.zeros((batchSize, maxLen, sig[0].shape[-1]), dtype=numpy.float32)
    for idx, s in enumerate(sig):
        sigPadded[idx,:s.shape[0]] = s
    return sigPadded, sigLen, sigmask, sigLabel



"""
For testing, the only difference exists in sampler
"""

class batchSampler_test(object):
    """docstring for sampler"""
    def __init__(self, dataset: dataset):
        super(batchSampler_test, self).__init__()
        self.numIters = len(dataset)
        self.index = numpy.arange(0, self.numIters, dtype=numpy.int32)
        self.accumNum = dataset.accumNum
        self.n_samples = self.numIters * self.accumNum
        self.numGen = dataset.numGen
        self.numNeg = dataset.numNeg

    def __iter__(self):
        for uidx in self.index:
            # One batch one user
            batch = numpy.arange(self.accumNum[uidx], 
                                 self.accumNum[uidx]+self.numGen[uidx]+self.numNeg[uidx], 
                                 dtype=numpy.int32)
            yield batch
        # for batch in range(self.n_samples):
        #     # One batch one sample
        #     yield numpy.array([batch]).reshape(1,)

    def __len__(self):
        return self.numIters
