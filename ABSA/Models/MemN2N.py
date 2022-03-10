import torch, time
import torch.nn as nn
from utils.Attention import Attentions
from utils.RNNs import BiLSTM


class MemN2N(nn.Module):

    def __init__(self, args, Data):
        super(MemN2N, self).__init__()
        self.score_dev = 0
        self.score_test = 0
        self.hops   = 1
        self.name   = 'MemN2N'
        self.base_type= args.base_model
        self.method = args.model_method
        self.args   = args
        self.embed  = nn.Embedding.from_pretrained(torch.tensor(Data.word_embed), freeze=True) # 预训练词向量(freeze=True冻住词向量)
        self.dropout  = nn.Dropout(args.drop_rate)

        self.bi_lstm  = BiLSTM(Data.embed_dim, Data.embed_dim)
        self.x_linear = nn.Linear(Data.embed_dim, args.embed_dim)
        self.relu     = nn.ReLU()
        self.attn     = Attentions(args.embed_dim, method='mlp') 
        self.tanh     = nn.Sigmoid() 
        self.dense    = nn.Linear(args.embed_dim, Data.n_class)

    def forward(self, inputs):

        seqs_id, ctxs_mask, seqs_len = inputs['seqs_id'], inputs['ctxs_mask'], inputs['seqs_len']
        asps_mask, asps_len = inputs['asps_mask'], inputs['asps_len']

        seqs = self.dropout(self.embed(seqs_id))
        # seqs, _ = self.bi_lstm(seqs, seqs_len)

        asps = torch.mul(seqs.permute(2,0,1), asps_mask).permute(1,2,0)
        asps = torch.div(torch.sum(asps, dim=1), asps_len)

        x = asps.unsqueeze(dim=1)
        for _ in range(self.hops): 
            x = self.x_linear(x) 
            out, score = self.attn(x, seqs, mask=ctxs_mask) 
            x = out + x
        
        out = self.dense(x)

        return out.squeeze(dim=1), score.squeeze(dim=1)

