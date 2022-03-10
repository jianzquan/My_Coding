import torch, math
import torch.nn as nn
from transformers import BertModel


class MyBert(nn.Module):

    def __init__(self, args, Data):
        super(MyBert, self).__init__()
        self.score_dev  = 0
        self.score_test = 0
        self.hidden_dim = 1024
        self.name   = 'MyBert'
        self.method = args.model_method
        self.args   = args
        self.bert   = BertModel.from_pretrained(args.pretrain)
        self.l_attn = nn.Linear(self.hidden_dim, 1)
        self.dense  = nn.Linear(self.hidden_dim, Data.n_class)
        self.dropout   = nn.Dropout(args.drop_rate)
        
    def forward(self, inputs):

        seqs_id, seqs_mask = inputs['seqs_id'], inputs['seqs_mask']
        out_bert = self.bert(seqs_id, attention_mask=seqs_mask)[0]

        asps_mask_raw = inputs['asps_mask']
        asps_mask = (1.0-asps_mask_raw) * -1e8
        asps_attn = self.l_attn(out_bert).squeeze(dim=-1)
        asps_attn_mask = torch.softmax(asps_attn+asps_mask, dim=-1)
        out_asps = torch.sum(out_bert*asps_attn_mask.unsqueeze(dim=-1), dim=1)
        
        sent_mask_raw = inputs['sent_mask']
        sent_mask = (1.0-sent_mask_raw+asps_mask_raw) * -1e8
        seqs_attn = torch.sum(out_bert*out_asps.unsqueeze(dim=1), dim=-1)/math.sqrt(self.hidden_dim)
        seqs_attn_mask = torch.softmax(seqs_attn+sent_mask, dim=-1)
        out_seqs = torch.sum(out_bert*seqs_attn_mask.unsqueeze(dim=-1), dim=1)

        out = self.dense(self.dropout(out_asps+out_seqs))

        return out, seqs_attn_mask

