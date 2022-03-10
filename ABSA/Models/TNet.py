import torch
import torch.nn as nn
from tqdm.std import tqdm
from utils.RNNs import BiLSTM
from utils.Attention import Attentions
from utils.CNNs import CNN


class CPT(nn.Module):
    def __init__(self, input_dim, n_layer=2, flag="LF"):
        super(CPT, self).__init__()
        self.input_dim = input_dim
        self.LF_AS     = flag  # 选择是 LF / AS

        self.attn      = Attentions(self.input_dim)
        self.linear    = nn.Linear(self.input_dim*2, self.input_dim)
        self.relu      = nn.ReLU()

        self.linear_as = nn.Linear(self.input_dim, self.input_dim)
        self.sigmoid   = nn.Sigmoid()

        
    def forward(self, contexts, querys, mask=None):

        attn_out, attn_score = self.attn(Q=querys, K=contexts, mask=mask)
        combine = torch.cat([attn_out, querys], dim=-1)
        linear_out = self.relu(self.linear(combine))

        if self.LF_AS == 'LF':
            cpt_out = querys + linear_out
        if self.LF_AS == 'AS':
            scale = self.sigmoid(self.linear_as(querys))
            cpt_out = torch.mul(scale, linear_out) + torch.mul(1-scale, querys)
        
        return cpt_out


class TNet(nn.Module):

    def __init__(self, args, Data):
        super(TNet, self).__init__()
        self.score_dev  = 0
        self.score_test = 0
        self.name       = 'TNet'
        self.method     = 'no'
        self.args       = args
        self.base_type  = args.base_model

        self.hidden_dim = Data.embed_dim
        self.embed      = nn.Embedding.from_pretrained(torch.tensor(Data.word_embed), freeze=True)  # 预训练词向量(freeze=True冻住词向量)
        self.dropout    = nn.Dropout(args.drop_rate)

        self.bi_lstm    = BiLSTM(Data.embed_dim, self.hidden_dim)
        self.cpt_layers = 2
        self.cpt        = CPT(self.hidden_dim, flag='AS')
        self.cnn_mask   = False
        self.cnn        = CNN(self.hidden_dim)
        self.attn       = QKVAttention(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.dense      = nn.Linear(self.hidden_dim, Data.n_class)


    def forward(self, inputs):

        seqs = self.dropout(self.embed(inputs['seqs_id']))
        seqs_lstm, _ = self.bi_lstm(seqs, inputs['seqs_len'])
        
        max_asp_len, asps_pad, masks = torch.max(inputs['asps_len']), [], []
        for i, asp_len in enumerate(inputs['asps_len']):
            item = seqs_lstm[i][inputs['asps_mask'][i]==0]
            add_item = torch.zeros(max_asp_len-asp_len, seqs_lstm.shape[-1])
            asps_pad.append(torch.cat([item, add_item]))
            mask = torch.ones(max_asp_len)*-1e8; mask[0:asp_len] = 0
            masks.append(mask)
        asps_pad = torch.stack(asps_pad); masks = torch.stack(masks)
        asps     = torch.div(torch.sum(asps_pad, dim=1), inputs['asps_len']).unsqueeze(dim=1)

        # max_seq_len, seqs_pad = torch.max(inputs['seqs_len']), []
        # for i, asp_len in enumerate(inputs['asps_len']):
        #     item = seqs_lstm[i][inputs['asps_mask'][i]<0]
        #     add_item = torch.zeros(asp_len, seqs_lstm.shape[-1])
        #     seqs_pad.append(torch.cat([item, add_item]))
        # seqs_pad = torch.stack(seqs_pad)

        x = seqs_lstm
        for _ in range(self.cpt_layers):
            x = self.cpt(seqs_lstm, x, mask=inputs['asps_mask'])

        # 2. convolution Layer
        cpt_out = x
        if self.cnn_mask:
            cpt_out = torch.mul(inputs['seqs_pos'].unsqueeze(dim=-1), cpt_out)
            out = self.cnn(cpt_out)
            score = inputs['seqs_pos'].unsqueeze(dim=-1).permute(0,2,1)
        else:
            out, score = self.attn(Q=asps, K=cpt_out, mask=inputs['sent_mask'])

        return self.dense(out.squeeze(dim=1)), score.squeeze(dim=1)
