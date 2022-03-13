import torch, math, copy
import torch.nn as nn
from torch.utils.data import random_split
from transformers import BertModel, BertTokenizer
import numpy as np

class MyBert(nn.Module):

    def __init__(self, args, Data):
        super(MyBert, self).__init__()
        self.changeArgs(args)
        self.score_dev  = 0
        self.score_test = 0
        self.hidden_dim = 768
        self.name   = 'MyBert'
        self.method = args.model_method
        self.args   = args
        self.bert   = BertModel.from_pretrained(args.pretrain)
        self.token  = BertTokenizer.from_pretrained(args.pretrain)
        self.base_type = args.base_model
        self.changeData(Data) # 重建Bert输入

        self.l_attn = nn.Linear(self.hidden_dim, 1)
        self.dense  = nn.Linear(self.hidden_dim, Data.n_class)
        self.dropout   = nn.Dropout(args.drop_rate)

    def changeArgs(self, args):
        args.batch_size = 64
        args.max_grad_norm = 1.0
        args.lr = 5e-5
        args.gamma = 0.1
        args.optim_type = 'AdamW'
        
        
    def changeData(self, vocab):
        vocab.train, vocab.val, vocab.test = None, None, None
        for key, data_all in vocab.datas.items():
            if data_all is None: continue
            data, temp = [], np.zeros(vocab.maxSeqLen*2+3).astype('int32')
            for i, item in enumerate(data_all):
                item['index'] = i
                sent = ' '.join(item['seq'])
                sent = sent + ' [SEP] ' + ' '.join(item['asp'])

                item['bert']   = self.token(sent)
                item['bert']['token_type_ids'][item['asp_pos'][0]+1:item['asp_pos'][-1]+1] = [1]*item['asp_len']
                item['bert']['token_type_ids'][-1-item['asp_len']:-1] = [1]*item['asp_len']

                # padding
                samp = {'indexes': 0, 'labels': 0, 'asps_len': 0, 'seqs_len': 0, 'seqs_id': copy.deepcopy(temp), 'seqs_mask': copy.deepcopy(temp), 'asps_mask': copy.deepcopy(temp), 'ctxs_mask': copy.deepcopy(temp)}
                samp['indexes']  = item['index']
                samp['labels']   = item['label']
                samp['seqs_len'] = len(item['bert']['input_ids'])
                samp['asps_len'] = item['asp_len']
                samp['seqs_id'][0:samp['seqs_len']]   = item['bert']['input_ids']
                samp['seqs_mask'][0:samp['seqs_len']] = item['bert']['attention_mask']
                samp['asps_mask'][0:samp['seqs_len']] = item['bert']['token_type_ids']
                samp['ctxs_mask'] = [val for val in samp['seqs_mask']]
                samp['ctxs_mask'] -= samp['asps_mask'] 
                samp['ctxs_mask'][0] = 0
                samp['ctxs_mask'][samp['seqs_len']-item['asp_len']-2:] = 0

                data.append(samp)
            
            if key == 'train': vocab.train = data
            if key == 'test':  vocab.test  = data
            if key in ['dev', 'val']: vocab.val = data
        
        if vocab.val is None and vocab.args.val_ratio>0: 
            val_len = int(len(vocab.train)*vocab.args.val_ratio)
            train_len = len(vocab.train) - val_len
            vocab.train, vocab.val = random_split(vocab.train, [train_len, val_len])

        
    def forward(self, inputs):

        seqs_id, seqs_mask = inputs['seqs_id'], inputs['seqs_mask']
        out_bert = self.bert(seqs_id, attention_mask=seqs_mask)[0]

        asps_mask_raw = inputs['asps_mask']
        asps_mask = (1.0-asps_mask_raw) * -1e8
        asps_attn = self.l_attn(out_bert).squeeze(dim=-1)
        asps_attn_mask = torch.softmax(asps_attn+asps_mask, dim=-1)
        out_asps = torch.sum(out_bert*asps_attn_mask.unsqueeze(dim=-1), dim=1)
        
        sent_mask_raw = inputs['ctxs_mask']
        sent_mask = (1.0-sent_mask_raw+asps_mask_raw) * -1e8
        seqs_attn = torch.sum(out_bert*out_asps.unsqueeze(dim=1), dim=-1)/math.sqrt(self.hidden_dim)
        seqs_attn_mask = torch.softmax(seqs_attn+sent_mask, dim=-1)
        out_seqs = torch.sum(out_bert*seqs_attn_mask.unsqueeze(dim=-1), dim=1)

        out = self.dense(self.dropout(out_asps+out_seqs))

        return out, seqs_attn_mask

