import copy
import torch
import torch.nn as nn


# 多分类
class NLLLoss_(nn.Module):

    def __init__(self, reduction='mean'):
        super(NLLLoss_, self).__init__()
        self.m    = nn.LogSoftmax()
        self.loss = nn.NLLLoss(reduction=reduction)

    def forward(self, input, target):
        # input : [batch_size, nclass] 
        # target: [batch_size, 1]
        # self.m(input): 对input计算softmax(每行和为1),区间(0,1); 然后取对数，区间(-inf,0) 
        # self.loss(-, target): 根据target取出 - 对应位置元素, 取均值返回. (相当于先将target转onehot, 然后与 - 点乘, 取平均值)
        return self.loss(self.m(input), target)

class CrossEntropyLoss_(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss_, self).__init__()
        # CrossEntropyLoss = LogSoftmax + NLLLoss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        # input : [batch_size, nclass] 
        # target: [batch_size, 1]
        # self.loss(input, target): 先对input计算logsoftmax,再按target取出每行对应位置元素，取平均值
        return self.loss(input, target)


# 二分类
class BCELoss_(nn.Module):
    # 二分类
    def __init__(self):
        super(BCELoss_, self).__init__()
        self.m    = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, input, target):
        # input : [batch_size]
        # target: [batch_size], 需要float类型
        # self.m(input): 归一化
        # self.loss(-, target): 对于每一个sample = y*In(y_)+(1-y)*In(1-y_), 求和取平均
        return self.loss(self.m(input), target)

class BCEWithLogitsLoss_(nn.Module):
    
    def __init__(self):
        super(BCEWithLogitsLoss_, self).__init__()
        # BCEWithLogitsLoss = Sigmoid + BCELoss
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        # input : [batch_size]
        # target: [batch_size], 需要float类型
        # self.loss(-, target): 先对input进行sigmoid归一化, 然后对于每一个sample = y*In(y_)+(1-y)*In(1-y_), 求和取平均
        return self.loss(input, target)


# 其他
class L1Loss_(nn.Module):
    # L1 Norm
    def __init__(self):
        super(L1Loss_, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        # input: [1, N]
        # target: 与input一样
        # self.loss(input, target): 计算对应元素差的绝对值，然后求和或取平均
        return self.loss(input, target)

class MSELoss_(nn.Module):
    # L2 Norm
    def __init__(self):
        super(MSELoss_, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        # input: [1, N]
        # target: 与input一样
        # self.loss(input, target): 计算对应元素差的平方，然后求和或取平均
        return self.loss(input, target)


class Score_(nn.Module):
    """
    Precision Recall F1-Score Micro-F1 Macro-F1

    TP: True  Positive 预测为1,实际为1 √
    FP: False Positive 预测为1,实际为0
    FN: False Negative 预测为0,实际为1
    TN: True  Negative 预测为0,实际为0 √

    P  = TP / (TP + FP)
    R  = TP / (TP + FN)
    F1 = 2*(P*R) / (P+R)

    example:
      data: 1 2 3 4 5 6 7 8 9
    target: A A A A B B B C C 
      pred: A A B C B B C B C

    for A: TP=2; FP=0; FN=2; TN=5;  P=2/(2+0); R=2/(2+2); F1=2(1*0.5)/(1+0.5)
    for B: TP=2; FP=2; FN=1; TN=4;  P=2/(2+2); R=2/(2+1); F1=2(0.5*0.66)/(0.5+0.66)
    for C: TP=1; FP=2; FN=1; TN=5;  P=1/(1+2); R=1/(1+1); F1=2(0.33*0.5)/(0.33+0.5)
      all: TP=5; FP=4; FN=4; TN=14; P=5/(5+4); R=5/(5+4); 
    
    Micro-F1 = 2*(P*R)/(P+R)   # (P=R=F1)
    Macro-F1 = (F1_A + F1_B + F1_C) / 3
    """
    def __init__(self, method='macro'):
        super(Score_, self).__init__()
        self.m    = nn.LogSoftmax(dim=-1)
        self.method = method

    def forward(self, inputs, targets):
        # 消除最后一维
        if inputs.shape != targets.shape:
            preds = torch.argmax(self.m(inputs), dim=-1) 
        else: preds = inputs
        # 消除 batch
        pred, target = preds.reshape(-1), targets.reshape(-1)
        label = set(target.cpu().tolist())
        label_dict = {int(key): 0 for key in label}
        
        TP, FP, FN = label_dict.copy(), label_dict.copy(), label_dict.copy()
        for pre, tar in zip(pred, target):
            # TP: pre=1, tar=1
            if pre == tar: 
                TP[int(tar.cpu().item())] += 1
            # FP: pre=1, tar=0; FN: pre=0, tar=1
            if pre != tar: 
                FP[int(pre.cpu().item())] += 1 
                FN[int(tar.cpu().item())] += 1

        if self.method == 'micro':
            total_TP = sum(list(TP.values()))
            total_FP = sum(list(FP.values()))
            total_FN = sum(list(FN.values()))
            P  = total_TP / (total_TP + total_FP + 1e-31)
            R  = total_TP / (total_TP + total_FN + 1e-31)
            F1 = 2*(P*R) / (P+R+1e-31) 
            return {'P': P, 'R': R, 'F1': F1}
        if self.method == 'macro':
            P, R, F1 = {}, {}, {}
            for key in label_dict.keys():
                P[key] = TP[key] / (TP[key]+FP[key]+1e-31)
                R[key] = TP[key] / (TP[key]+FN[key]+1e-31)
                F1[key] = 2*(P[key]*R[key]) / (P[key]+R[key]+1e-31) 
            return{'P':  sum(list(P.values()))/len(label_dict),
                   'R':  sum(list(R.values()))/len(label_dict),
                   'F1': sum(list(F1.values()))/len(label_dict),}
                






