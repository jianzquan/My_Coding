
import torch, random, copy, pickle, os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils.Processor import Processor, initial_params
from Models.TNet import TNet
from Models.MemN2N import MemN2N
from Models.MyBert import MyBert
e = torch.tensor(1e-32)


def a_loss(self, batch, preds, scores):
    
    indexes, labels, Attns = batch['indexes'], batch['labels'], self.Attn
    preds = torch.argmax(F.log_softmax(preds, dim=-1), dim=-1)
    seqs_len, asps_len  = batch['seqs_len'], batch['asps_len']    
    
    loss = 0
    for i in range(len(labels)):
        attn = Attns[indexes[i]][0:seqs_len[i]] # 优化 attn
        sent = self.data.datas['train'][indexes[i]]['seq']
        score = scores[i].detach().cpu().numpy()[0:seqs_len[i]] # 当前 score
        assert score.shape == attn.shape

        # 查找一阶显著词
        stand_0 = min(1.5/(seqs_len[i]-asps_len[i]), 1) # 重要词阈值
        masks_0 = score >= stand_0.item() # score 中是否有那个词权重大于阈值
        # reduc_0 = 0.5 / sum(score>0) # 不重要词阈值
        # maskc_0 = (score <= reduc_0)   # score 中是否有那个词权重小于阈值(排除asp)

        if preds[i] == labels[i]:
            attn += score*masks_0
            # expect[maskc1] = 0
        else: attn -= score*masks_0

        if sum(attn) > 0:
            # 查找二阶显著词
            stand_1 = min(2 * np.sum(attn)/sum(attn>0), np.sum(attn)) # attn 中重要词阈值
            masks_1 = attn >= stand_1 # attn 中是否有词异常重要
            if sum(masks_1) == 0:
                # 重要的词差距不大，仅对最大的进行偏向
                loss += (1-score[attn.argmax()])**2 
            else:
                # 重要的词差距较大，对更重要的词进行偏向
                temp = [1/sum(masks_1) if val>stand_1 else 0 for val in attn]
                loss += np.sum((temp-score*masks_1)**2)

    return loss /len(labels)


def Train_AT(self, desc='train'):

    temp = np.zeros_like(self.data.train.dataset[0]['seqs_mask']).astype('float32')
    self.Attn = [copy.deepcopy(temp) for _ in self.data.train.dataset]

    print(">>> {}_{}, lr_{}, batch_size_{}".format(self.data.name, self.model.name+'_'+self.model.method, self.args.lr, self.args.batch_size))
    
    # 0. import base model
    self.importModel()

    # 1. training
    lossRec = np.array([])
    while self.early_stop <= self.args.early_stop: 
        loss_train = self.oneEpoch(other=a_loss)
        lossRec = np.append(lossRec, loss_train)
        # self.changeLr(lossRec, threshold=2)

