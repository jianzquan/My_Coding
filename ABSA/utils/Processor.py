import torch, math, pickle, os, time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=log
from utils.LossFunc import Score_


def initial_params(model): 
    for child in model.children():
        for param in child.parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    torch.nn.init.xavier_uniform_(param) # 1维的用不了
                else:
                    stdv = 1. / math.sqrt(param.shape[0])
                    torch.nn.init.uniform_(param, a=-stdv, b=stdv)
            else:
                param.requires_grad = True
                pass


class Criterion(object):
    def __init__(self) -> None:
        super(Criterion, self).__init__()  
        self.loss_f = nn.CrossEntropyLoss()
        self.evaluate = Score_(method='macro')

    def loss(self, preds, truths):

        pred, target = preds, truths['labels']
        loss = self.loss_f(pred, target.to(torch.long))

        return loss

    def score(self, preds, truths, desc='f1'):

        pred, target = preds, truths['labels']
        score = self.evaluate(pred, target)

        return {'f1': score['F1']*100, 'acc': score['P']*100}
        # return {'f1_score': score['F1']}

class Processor(object):

    def __init__(self, args, data, model) -> None:
        super(Processor, self).__init__()
        self.args    = args
        self.model   = model
        self.data    = data
        self.epoch   = 0
        self.early_stop = 0
        self.score_dev  = {'f1': 0, 'acc': 0}
        self.criterion  = Criterion()
        self.optimizer  = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0)
        # self.tb_writer  = SummaryWriter(log_dir=args.tb_dir)

    def squeezePadding(self, batch):

        seqs_id, seqs_len = batch['seqs_id'], batch['seqs_len']
        seqs_mask, asps_mask, ctxs_mask = batch['seqs_mask'], batch['asps_mask'], batch['ctxs_mask']

        seqs_id_pad   = pack_padded_sequence(seqs_id, seqs_len, batch_first=True, enforce_sorted=False)
        seqs_mask_pad = pack_padded_sequence(seqs_mask, seqs_len, batch_first=True, enforce_sorted=False)
        ctxs_mask_pad = pack_padded_sequence(ctxs_mask, seqs_len, batch_first=True, enforce_sorted=False)
        asps_mask_pad = pack_padded_sequence(asps_mask, seqs_len, batch_first=True, enforce_sorted=False)

        batch['seqs_id']   = pad_packed_sequence(seqs_id_pad, batch_first=True)[0].to(torch.long)
        batch['seqs_mask'] = pad_packed_sequence(seqs_mask_pad, batch_first=True)[0]
        batch['ctxs_mask'] = pad_packed_sequence(ctxs_mask_pad, batch_first=True)[0]
        batch['asps_mask'] = pad_packed_sequence(asps_mask_pad, batch_first=True)[0]
        
        batch['asps_len'] = batch['asps_len'].unsqueeze(dim=1)
        batch['seqs_len'] = batch['seqs_len'].unsqueeze(dim=1)

        return batch

    def changeLr(self, lossRec, threshold=3):
   
        if len(lossRec)>threshold and all(lossRec[-threshold]<=lossRec[-threshold:]):
            self.args.lr *= 0.9
        elif len(lossRec)>threshold and lossRec[-2]>lossRec[-1]*1.02:
            return -1
            # self.args.lr *= 1.1
        else:
            return -1
        print('learning_rate: {:.2f}'.format(self.args.lr))
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer  = optim.SGD(params, lr=self.args.lr, weight_decay=0)

    def importModel(self):
        
        file = ".\\Models\\base\\{}_{}_{}".format(self.data.name, self.model.name, self.model.method)
        if self.args.base_model>0 and os.path.exists(file):
            model = torch.load(file)
            self.model = model
            self.model.base_type = self.args.base_model
            self.model.args = self.args
            self.model.dropout = nn.Dropout(self.model.args.drop_rate)
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr, weight_decay=0)
            if self.model.base_type == 2: # fix > score_base
                self.model.base_type = 20
            if self.model.base_type == 3: # update always
                self.args.score_base = model.score_dev
            score_train, score_dev = self.test(desc='dev')
            self.model.score_dev = score_dev
            print(">> initial, train_score: {}, dev_score: {}".format(score_train, score_dev))

    def oneEpoch(self, other=None):
        begin_time = time.time()
        self.early_stop += 1; self.epoch += 1; loss_train = 0
        dataLoader = DataLoader(self.data.train, self.args.batch_size, shuffle=True)   
        self.model.train()
        for i, batch in enumerate(dataLoader):
            self.optimizer.zero_grad()
            batch = self.squeezePadding(batch)
            pred, attn = self.model(batch)
            loss  = self.criterion.loss(pred, batch)
            if other is not None: loss += other(self, batch, pred, attn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            loss_train += loss.item()*self.args.batch_size
            
        score_train, score_dev = self.test(desc='dev')
        end_time = time.time()
        print('> epoch {}, train_loss: {:.2f}, train_score: {:.2f}, dev_score: {:.2f}, time: {:.2f}'.format(self.epoch, loss_train, score_train, score_dev, end_time-begin_time))
        
        return loss_train

    def train(self, desc='train'):
        print(">>> {}_{}, lr_{}, batch_size_{}".format(self.data.name, self.model.name+'_'+self.model.method, self.args.lr, self.args.batch_size))
        
        # 0. import base model
        self.importModel()

        # 1. training
        lossRec = np.array([])
        while self.early_stop <= self.args.early_stop: 
        # while self.epoch <= self.args.epochs:
            loss_train = self.oneEpoch()
            lossRec = np.append(lossRec, loss_train)
            # self.changeLr(lossRec, threshold=2)

        #     # 写入tensorboard
        #     if self.epoch == 1: self.tb_writer.add_graph(self.model, (inputs, ))
        #     self.tb_writer.add_scalar("loss_train", loss_train, self.epoch)
        #     self.tb_writer.add_scalar("f1_train", score_train, self.epoch)
        #     self.tb_writer.add_scalar("f1_dev",   score_dev, self.epoch)
            

        # self.tb_writer.close()

    def test(self, desc='test'):
        self.model.eval()

        # 0. import data
        if desc == 'dev':
            dataLoader_0 = DataLoader(self.data.train, len(self.data.train))
            if self.args.val_ratio < 0.2:
                dataLoader_1 = DataLoader(self.data.test, len(self.data.test))
            else:
                dataLoader_1 = DataLoader(self.data.val, len(self.data.val.indices))

        if desc == 'test':
            if self.args.val_ratio < 0.2:
                dataLoader_0 = DataLoader(self.data.test, len(self.data.test))
            else:
                dataLoader_0 = DataLoader(self.data.val, len(self.data.val.indices))
            dataLoader_1 = DataLoader(self.data.test, len(self.data.test))

        # 1. calculate scores
        with torch.no_grad():
            for batch_0, batch_1 in zip(dataLoader_0, dataLoader_1):
                batch_0   = self.squeezePadding(batch_0)
                pred_0, _ = self.model(batch_0)
                score_0   = self.criterion.score(pred_0, batch_0, desc='f1_score')

                batch_1   = self.squeezePadding(batch_1)
                pred_1, _ = self.model(batch_1)
                score_1   = self.criterion.score(pred_1, batch_1, desc='f1_score')  

        # 2. update global param
        keys = list(score_1.keys())  # score 共有几项
        # 2.1 for dev
        if desc == 'dev':
            # processor score
            if self.score_dev[keys[0]] < score_1[keys[0]]:
                self.score_dev[keys[0]] = score_1[keys[0]] 
                self.early_stop = 0
            # model score
            if self.model.score_dev < score_1[keys[0]]:
                self.model.score_dev = score_1[keys[0]] 
                torch.save(self.model, self.args.output_dir)
                print(">> {} update: {}".format(self.args.output_dir, self.model.score_dev))
            if not os.path.exists(self.args.output_dir):
                from shutil import copyfile
                source = ".\\Models\\base\\{}_{}_{}".format(self.data.name, self.model.name, self.model.method)
                target = self.args.output_dir
                copyfile(source, target)
            # base model
            if self.model.base_type>0 and self.model.base_type<4 and score_1[keys[0]]>=self.args.score_base:
                if self.model.base_type == 1: # fix = score_base (save one times)
                    self.model.base_type = 10
                file = ".\\Models\\base\\{}_{}_{}".format(self.data.name, self.model.name, self.model.method)
                torch.save(self.model, file)
        # 2.2 for test
        if desc == 'test':
            self.model.score_test = score_1[keys[0]]
            torch.save(self.model, self.args.output_dir+'_'+str(round(score_1['f1'], 2)))


        return score_0['f1'], score_1['f1']
            




