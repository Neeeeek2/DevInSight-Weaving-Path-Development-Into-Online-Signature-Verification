#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy
import random
import pdb
import sys
from torch.functional import block_diag
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torchvision.utils as vutils
from torch.autograd import Variable

from DevNet.development.nn import development_layer
from DevNet.development.so import so
from DevNet.development.sp import sp
from DevNet.development.se import se
from DevNet.development.hyperbolic import hyperbolic
from DevNet.development.unitary import unitary

from soft_dtw_cuda import SoftDTW

from Everyday_Attention import *


class RNN_glb_integ(nn.Module):
    def __init__(self, 
                 in_dim, 
                 n_layers=1,
                 bidirectional=False,
                 ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        # self.rnn = nn.LSTM(in_dim, in_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        self.rnn = nn.GRU(in_dim, in_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        self.GA = Gated_Addition(in_dim)
    
    def forward(self, x, mask):
        input = x
        x = self.ln(x)
        x = nn.utils.rnn.pack_padded_sequence(
            x, mask.sum(1).long().cpu(), 
            batch_first=True, enforce_sorted=False
            )
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # x = input + x
        x = self.GA(input, x)
        return x


class pthdv_block(nn.Module):
    def __init__(self, 
                 in_dim = 64, 
                 out_dim = 64,
                 pthdv_m = 8,
                 kernel = 2,
                 stride = 1,
                 param = so,
                 ) -> None:
        super(pthdv_block, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.m = pthdv_m
        
        self.n_groups = out_dim // pthdv_m**2 * 4
        self.pthdv_layer = development_layer(
            in_dim//self.n_groups, 
            pthdv_m, 
            return_sequence=False, 
            param=param, 
            triv=torch.matrix_exp
            )
        self.ln_dev1 = nn.LayerNorm(pthdv_m**2)
        self.proj = nn.Linear(self.n_groups*pthdv_m**2, out_dim)

        # SE
        self.se = Squeeze_n_Excitation(out_dim, 16, nn.GELU)

        # residual connection for the whole block
        self.res_con = nn.ModuleList([
            nn.Identity() if in_dim==out_dim else nn.Linear(in_dim,out_dim), 
            nn.Identity() if stride==1 else nn.MaxPool1d(stride, stride)
            ])

    def forward(self, x, mask):
        '''
        `x`: tensor of shape (B,T,D)\n
        `mask`: tensor of shape (B,T)
        '''
        B,T,D = x.shape
        input = x

        ## path development layer
        feat_dev = []
        group_dim = D // self.n_groups
        for i in range(self.n_groups):
            x_group = x[:,:,i*group_dim:(i+1)*group_dim]  # (B,T,group_dim)
            feat_dev_1 = self.pthdv_layer(
                x_group, kernel=self.kernel, stride=self.stride, mask=mask
                )  # (B,T',1,m,m)
            feat_dev_1 = torch.flatten(feat_dev_1, -3)  # (B,T',m**2)
            feat_dev_1 = self.ln_dev1(feat_dev_1)
            feat_dev.append(feat_dev_1)
        feat_dev = torch.concat(feat_dev, dim=-1)
        feat_dev = self.proj(feat_dev)
        ## resample T' to T
        tar_T = T // self.stride
        feat_dev = F.interpolate(feat_dev.permute(0,2,1), size=tar_T, mode='linear', align_corners=False).permute(0,2,1)
        ## res add
        x = input+feat_dev

        # SE
        # mask = mask[:, 1:] * mask[:, :-1]
        x = self.se(x,mask)  # (B,T,D)

        # residual connection for the whole block
        x_res = self.res_con[0](input)
        x_res = self.res_con[1](x_res.permute(0,2,1)).permute(0,2,1)
        x = x + x_res
        
        return x, mask


class network(nn.Module):
    def __init__(self, in_dim,
                n_classes, 
                n_task = 1,
                n_shot_g = 20, 
                n_shot_f = 20, # MSDS evaluate setting
                lie = 'sp',
                m = 8, 
                # Nmax = 480
                ):
        super(network, self).__init__() 
        ''' Define the network and the training loss. '''
        self.n_classes = n_classes
        self.n_task = n_task 
        self.n_shot_g = n_shot_g 
        self.n_shot_f = n_shot_f

        embd_dim = 64
        self.block_setting = [1,1,1]
        lie_lookup = {'so':so, 'se':se, 'sp':sp}

        ### stem layer
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, embd_dim, 7, 1, 3),
            nn.GELU(),
        )
        self.ln_stem = nn.LayerNorm(embd_dim)

        for i, n_blocks in enumerate(self.block_setting):
            for j in range(n_blocks):
                setattr(self, f'pthdv_block{i}{j}', 
                        pthdv_block(
                            embd_dim*(2**i), embd_dim*(2**i), 
                            pthdv_m=m,
                            kernel=2, stride=1,
                            param=lie_lookup[lie]
                            )
                        )
                setattr(self, f'seq_glb_integ{i}{j}', 
                        RNN_glb_integ(embd_dim*(2**i), 1, bidirectional=False)
                        )
        for i in range(len(self.block_setting)-1):
            setattr(self, f'ln_PM{i}', nn.LayerNorm(embd_dim*(2**i)))
            setattr(self, f'conv_PM{i}', nn.Conv1d(embd_dim*(2**i), embd_dim*(2**(i+1)), 2, 2))
        feat_dim = embd_dim*(2**(len(self.block_setting)-1))
        
        # dtw setting
        self.dtw = SoftDTW(True, gamma=1, normalize=False, bandwidth=0.1)
        # pooling to squeeze the time dimension
        # self.pooling_dtw = nn.AvgPool1d(2, 2)
        # self.pooling_dtw = nn.MaxPool1d(2, 2)

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        # params initialization
        for m in module.modules():
            if isinstance(m, nn.Conv1d):#如果是卷积层，参数kaiming分布处理
                nn.init.kaiming_normal_(m.weight, 1)
            elif isinstance(m, nn.LSTM):# 如果是lstm
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.uniform_(param,-0.1,0.1)  # 使用正态分布初始化权重
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)  # 使用常数初始化偏置项
            elif isinstance(m, nn.BatchNorm1d):#如果是批量归一化则伸缩参数为1，偏移为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):#如果是线性层，参数kaiming分布处理
                nn.init.kaiming_normal_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x:torch.tensor, mask:torch.tensor=None):
        '''
        x: tensor of shape (B,T,D)
        mask: tensor of shape (B,T)
        '''
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)

        x = self.stem(x.permute(0,2,1))  # (B,D,T)
        x = self.ln_stem(x.permute(0,2,1)) # (B,T,D)
        mask = nn.Upsample(size=x.shape[1], mode='nearest')(mask.unsqueeze(1)).squeeze(1)

        for i, n_blocks in enumerate(self.block_setting):
            for j in range(n_blocks):
                x, mask = getattr(self, f'pthdv_block{i}{j}')(x, mask)
                x = getattr(self, f'seq_glb_integ{i}{j}')(x, mask)

            if i < len(self.block_setting)-1:
                x = getattr(self, f'ln_PM{i}')(x)
                x = getattr(self, f'conv_PM{i}')(x.permute(0,2,1)).permute(0,2,1)
                mask = nn.Upsample(size=x.shape[1], mode='nearest')(mask.unsqueeze(1)).squeeze(1)

        return x, mask


    def tripletLoss_fnp(self, x, margin=0.25):
        step = self.n_shot_g + self.n_shot_f
        intra_g = intra_f = inter_fa = 0
        triLoss_std = triLoss_hard = 0
        triLoss_hard2 = 0

        anchors = list(x[i*step].unsqueeze(0) for i in range(self.n_task))
        anchors = torch.concat(anchors,dim=0)
        anchors = anchors / torch.norm(anchors, dim=1, keepdim=True)

        # x = F.relu(self.triplet_dense(x), inplace=True)
        for i in range(self.n_task):
            anchor = x[i*step]
            pos = x[i*step+1:i*step+self.n_shot_g]
            neg = x[i*step+self.n_shot_g:(i+1)*step]
            anchor = anchor / torch.norm(anchor)
            pos = pos / torch.norm(pos, dim=1, keepdim=True)
            neg = neg / torch.norm(neg, dim=1, keepdim=True)
            dist_g = torch.sum((anchor.unsqueeze(0) - pos)**2, dim=1)
            dist_f = torch.sum((anchor.unsqueeze(0) - neg)**2, dim=1)
            ### Inner class variation
            intra_g += torch.mean(dist_g)
            intra_f += torch.mean(dist_f)
            ### Triplet loss, self.n_shot_g * self.n_shot_f triplets in total
            triLoss = F.relu(dist_g.unsqueeze(1) - dist_f.unsqueeze(0) + margin) #(self.n_shot_g, self.n_shot_f)
            triLoss_std += torch.mean(triLoss) 
            triLoss_hard += torch.sum(triLoss) / (triLoss.data.nonzero().size(0) + 1) # batch hard sample mining
            # dist_g, _ = torch.sort(dist_g, descending=True)
            # dist_f, _ = torch.sort(dist_f, descending=True)
            # triLoss_hard += torch.mean(F.relu(dist_g[:2].unsqueeze(1) - dist_f[-4:].unsqueeze(0) + margin))
            idx_anc = list(set(range(self.n_task))-set([i]))
            dist_fa = torch.sum((anchors[idx_anc,:].unsqueeze(0) - neg.unsqueeze(1))**2, dim=2)
            triLoss2 = F.relu(dist_f.unsqueeze(1) - dist_fa)
            triLoss_hard2 += torch.sum(triLoss2) / (triLoss2.data.nonzero().size(0) + 1)
            ### Inter class variation
            inter_fa += torch.mean(dist_fa)

        intra_g = intra_g / self.n_task
        intra_f = intra_f / self.n_task
        inter_fa = inter_fa / self.n_task
        triLoss_std = triLoss_std / self.n_task
        triLoss_hard = triLoss_hard / self.n_task
        triLoss_hard2 = triLoss_hard2 / self.n_task

        return triLoss_hard, triLoss_std, triLoss_hard2, intra_g, intra_f, inter_fa

    def measurement_Loss(self, 
                         x: torch.FloatTensor, 
                         lens: torch.FloatTensor, 
                         margin: float=1.0, 
                         ):
        '''
        B is consist of n_task * (n_shot_g+n_shot_f)
        '''
        # x = F.normalize(x, p=2, dim=-1)
        
        '''
        For every user, we regard every genuine sample as anchor, (a)
        as the same, every genuine sample as positive sample, (g)
        every fake sample as negative sample, (sf)
        as well as samples from other users in the batch as negative samples (rf)
        '''
        Loss_tri = 0
        # Loss_tri_rf = 0
        intra_g = intra_f = inter_fa = 0
        step = self.n_shot_g + self.n_shot_f
        for u in range(self.n_task):
            idx_u = torch.arange(u*step, (u+1)*step)
            idx_g = idx_u[:self.n_shot_g]
            genu = x[idx_g]
            # anchor = genu
            anchor = genu[0].unsqueeze(0)
            genu = genu[1:]
            idx_n = idx_u[self.n_shot_g:]
            sfs = x[idx_n]
            # idx_rf = torch.arange(x.shape[0])[~torch.isin(torch.arange(x.shape[0]), idx_u)]
            idx_rf = torch.arange(self.n_task)[~torch.isin(torch.arange(self.n_task), u)]*step
            idx_rf = (idx_rf.unsqueeze(1) + torch.arange(2).unsqueeze(0)).flatten()
            rfs = x[idx_rf]
            #####
            lens_a = lens[idx_g[0:1]]
            lens_g = lens[idx_g[1:]]
            lens_sf = lens[idx_n]
            lens_rf = lens[idx_rf]
            
            ### soft-DTW distance
            # TODO: padded area of 0 would be no probelm in normal DTW, but could cause softmin in soft-dtw making greatly negtive values.
            '''x: (B,T,D)'''
            Na = len(anchor) ; Ng = len(genu) ; Nsf = len(sfs) ; Nrf = len(rfs)
            T_x, Dim_x = x.shape[1:]
            ## anchor VS genuine
            candi_a = anchor.unsqueeze(1).expand(-1, genu.shape[0], -1, -1).reshape(-1, T_x, Dim_x)
            candi_g = genu.unsqueeze(0).expand(anchor.shape[0], -1, -1, -1).reshape(-1, T_x, Dim_x)
            len_ag = lens_a.unsqueeze(1) + lens_g.unsqueeze(0)
            D_ap = self.dtw(candi_a, candi_g)       # (Na*Ng, )
            D_ap = D_ap.reshape(Na, Ng) / len_ag / Dim_x
            ## anchor VS skilled forgery
            candi_a = anchor.unsqueeze(1).expand(-1, sfs.shape[0], -1, -1).reshape(-1, T_x, Dim_x)
            candi_sf = sfs.unsqueeze(0).expand(anchor.shape[0], -1, -1, -1).reshape(-1, T_x, Dim_x)
            len_asf = lens_a.unsqueeze(1) + lens_sf.unsqueeze(0)
            D_asf = self.dtw(candi_a, candi_sf)     # (Na*Nsf, )
            D_asf = D_asf.reshape(Na, Nsf) / len_asf / Dim_x
            ## anchor VS random forgery
            candi_a = anchor.unsqueeze(1).expand(-1, rfs.shape[0], -1, -1).reshape(-1, T_x, Dim_x)
            candi_rf = rfs.unsqueeze(0).expand(anchor.shape[0], -1, -1, -1).reshape(-1, T_x, Dim_x)
            len_arf = lens_a.unsqueeze(1) + lens_rf.unsqueeze(0)
            D_arf = self.dtw(candi_a, candi_rf)     # (Na*Nrf, )
            D_arf = D_arf.reshape(Na, Nrf) / len_arf / Dim_x
            # TODO: sf VS rf, in parallel way

            ### triplet loss
            triplets = F.relu(D_ap.unsqueeze(2) - D_asf.unsqueeze(1) + margin) # (a,p,sf)
            # only select the positive triplets
            triplets =  triplets[triplets>0]
            # Loss_agsf = torch.log(
            #     1 + \
            #     torch.exp(triplets).sum()
            # )
            # Loss_agsf = triplets.sum() / (1 + torch.nonzero(triplets).size(0))

            # ### hard mining triplet loss
            # triplets = F.relu(D_ap_hard - D_asf_hard + margin) # (a,1)
            # triplets =  triplets[triplets>0]
            # Loss_agsf = torch.log(
            #     1 + \
            #     torch.exp(triplets).sum()
            # )

            # ### co-tuplet loss
            # triplets_ghard = F.relu(D_ap_hard - D_asf + margin) # (a,sf)
            # triplets_ghard = triplets_ghard[triplets_ghard>0]
            # triplets_sfhard = F.relu(D_ap - D_asf_hard + margin) # (a,rf)
            # triplets_sfhard = triplets_sfhard[triplets_sfhard>0]
            # Loss_agsf = torch.log(
            #     1 + \
            #     torch.exp(triplets_ghard).sum() + \
            #     torch.exp(triplets_sfhard).sum()
            # )
            
            ### triplet loss for rf samples
            # triplets_rf = F.relu(D_asf.unsqueeze(2) - D_sfrf.unsqueeze(0) + margin) # (a,sf,rf)
            triplets_rf = F.relu(D_ap.unsqueeze(2) - D_arf.unsqueeze(1) + margin) # (a,p,rf)
            triplets_rf =  triplets_rf[triplets_rf>0]
            # Loss_asfrf = torch.log(
            #     1 + \
            #     torch.exp(triplets_rf).sum()
            # )
            # Loss_asfrf = triplets_rf.sum() / (1 + torch.nonzero(triplets_rf).size(0))
            
            Loss_agsf = torch.log(
                1 + \
                torch.exp(triplets).sum() + \
                torch.exp(triplets_rf).sum()
            )
            # if torch.isinf(Loss_agsf): raise ValueError('Loss_agsf is inf')               

            Loss_tri += Loss_agsf / self.n_task
            # Loss_tri_rf += Loss_asfrf / self.n_task
            intra_g += D_ap.mean() / self.n_task
            intra_f += D_asf.mean() / self.n_task
            # inter_fa += D_sfrf.mean() / self.n_task
            inter_fa += D_arf.mean() / self.n_task
        return Loss_tri, intra_g, intra_f, inter_fa

    def smoothCEloss(self, logit, target, eps=0.1):
        n_class = logit.size(1)
        one_hot = torch.zeros_like(logit,dtype=logit.dtype).scatter(1, target.to(dtype=torch.int64).view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)
        return -(one_hot * log_prb).sum(dim=1).mean(dim=0)
        # return -((one_hot * log_prb).sum(dim=1) * self.smoothCElossMask).mean(dim=0)




if __name__=='__main__':
    B=4*(6+6) ; T=800 ; C=15
    model = network(C,202,4,6,6).cuda()
    x = torch.randn(B,T,C).cuda()
    mask =  torch.ones(B,T).cuda()
    # mask = torch.zeros(B,T).cuda()
    # for b in range(B):
    #     mask[b,:torch.randint(T//4,T,(1,))] = 1

    outputs = model(x,mask)
    # lens = (mask.sum(1)*outputs[1].shape[1]/T).to(dtype=torch.int32)
    # loss_tri, *_ = model.measurement_Loss(outputs[1], lens, margin=1.)
    # loss_tri.backward()
    print(f'Input shape: {x.shape}')
    for i, o in enumerate(outputs):
        print(f'Output{i} shape: {o.shape}')
    print()