import time
import pickle
import numpy 
import torch
import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from soft_dtw_cuda import SoftDTW
from fastdtw import fastdtw

from concurrent.futures import ProcessPoolExecutor


softdtw = SoftDTW(True, gamma=1e-16, normalize=False, bandwidth=.1)

def dis_1vs1(x, y, i, j, 
             way: str = "fastdtw"):
    if way == "fastdtw-cuda":
        # using cuda for dtw calculation
        dist_now = softdtw(
                torch.from_numpy(x).cuda().unsqueeze(0), 
                torch.from_numpy(y).cuda().unsqueeze(0)
                ).item()
    elif way == "fastdtw":
        # using cpu(numpy) for dtw calculation
        dist_now = fastdtw(x, y, radius=2)[0]
    elif way == "l2":
        # using l2-norm
        dist_now = numpy.linalg.norm(x-y)
    else:
        raise ValueError("Invalid way for dtw calculation")
    dist_now /= (len(x)+len(y))*x.shape[-1]     # norm with lengths & dimensions
    return dist_now, i, j

def dis_NvsN(seqs_x, seqs_y, 
             way: str = "fastdtw"):
    dist_matrix = numpy.zeros((len(seqs_x), len(seqs_y)))

    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(len(seqs_x)):
            for j in range(len(seqs_y)):
                x = seqs_x[i] ; y = seqs_y[j]
                futures.append(executor.submit(dis_1vs1, x, y, i, j, way))
        for future in futures:
            dist_now, i, j = future.result()
            dist_matrix[i, j] = dist_now
    
    return dist_matrix


def getEER(FAR, FRR):
    a = FRR <= FAR
    s = numpy.sum(a)
    a[-s-1] = 1
    a[-s+1:] = 0
    FRR1 = FRR[a]
    FAR1 = FAR[a] 
    
    a = [[FRR1[1]-FRR1[0], FAR1[0]-FAR1[1]], [-1, 1]]
    b = [(FRR1[1]-FRR1[0])*FAR1[0]-(FAR1[1]-FAR1[0])*FRR1[0], 0]
    return numpy.linalg.solve(a, b)

def scoreScatter(gen, forg):
    ax = plt.subplot()
    ax.scatter(gen[:,2],gen[:,1], color='r')
    ax.scatter(forg[:,2],forg[:,1], color='k', marker="*")
    # ax.set_xlim((0.0, 2.5))
    # ax.set_ylim((0.0, 2.5))
    ax.set_xlabel("dmin")
    ax.set_ylabel("dmean")
    ax.grid("on")
    k = (numpy.sum(gen[:,1] / gen[:,0]) + numpy.sum(forg[:,1] / forg[:,0])) / (gen.shape[0] + forg.shape[0])
    x = numpy.linspace(0, 0.3, 1000)  
    y = -x / k + 0.27
    plt.plot(x, y, 'k')
    plt.title("DISTANCE")
    plt.show()

def selectTemplate(distMatrix):
    refNum = distMatrix.shape[0]
    if refNum == 1:
        return None, 1, 1, 1, 1, 1
    # distMatrix = distMatrix + distMatrix.transpose()
    '''index of the template signature'''
    idx = numpy.argmin(numpy.sum(distMatrix, axis=1) / (refNum - 1))
    dvar = numpy.sqrt((numpy.sum(distMatrix**2) / refNum / (refNum - 1)- (numpy.sum(distMatrix) / refNum / (refNum - 1))**2))
    '''pair-wise distance'''
    dmean = numpy.sum(distMatrix) / refNum / (refNum - 1) 
    '''distance of reference signatures to the template signature dtmp'''
    dtmp = numpy.sum(distMatrix[:, idx]) / (refNum - 1)
    '''distance of reference signatures to their farthest neighbor dmax'''
    dmax = numpy.mean(numpy.max(distMatrix, axis=1))
    '''distance of reference signatures to their nearest neighbor dmin'''
    distMatrix[range(refNum), range(refNum)] = float("inf")
    distMatrix[distMatrix==0] = float("inf")
    dmin = numpy.mean(numpy.min(distMatrix, axis=1))
    
    return idx, dtmp**0.5, dmax**0.5, dmin**0.5, dmean**0.5, dvar
