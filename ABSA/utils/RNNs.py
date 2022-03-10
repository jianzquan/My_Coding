import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        pass



class BiLSTM(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.bi_lstm = nn.LSTM(in_feats, out_feats//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, seqs_len=None):

        batch_size = inputs.size()[0]
        # seqs_len -> tensor: [batch_size]
        # inputs -> tensor: [batch_size, seq_len, dimension]
        # ht: hidden state of each word -> [batch_size, seq_len, dimension]
        # hn: final forward vector and backword vector -> [2, batch_size, dimension/2] 
        
        if seqs_len is not None: # eliminate the effect of padding
            while len(seqs_len.shape) > 1: seqs_len = seqs_len.squeeze(dim=-1)
            inputs = pack_padded_sequence(inputs, seqs_len, batch_first=True, enforce_sorted=False)
            ht, (hn, cn) = self.bi_lstm(inputs)
            ht, _ = pad_packed_sequence(ht, batch_first=True)
        else:
            ht, (hn, cn) = self.bi_lstm(inputs)

        # assert batch_size == hn.size()[1], 'lstm input type is wrong'
        return self.dropout(ht), hn.permute(1,0,2).reshape(batch_size, -1)
