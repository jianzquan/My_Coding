import torch, math
import torch.nn as nn


class Attentions(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, n_head=1, drop_rate=0.1, method='mlp') -> None:
        super(Attentions, self).__init__()
        if hidden_dim is None: hidden_dim = input_dim // n_head
        if output_dim is None: output_dim = input_dim

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_head     = n_head
        self.method     = method

        self.w_q = nn.Linear(input_dim, hidden_dim*n_head)
        self.w_k = nn.Linear(input_dim, hidden_dim*n_head)
        self.w_v = nn.Linear(input_dim, hidden_dim*n_head)
        self.dropout = nn.Dropout(drop_rate)
        self.dense   = nn.Linear(hidden_dim*n_head, output_dim)
        
        stdv = 1. / math.sqrt(hidden_dim)
        if method == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
            self.weight.data.uniform_(-stdv, stdv)
        if method == 'linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            self.weight.data.uniform_(-stdv, stdv)


    def forward(self, q, k, v=None, mask=None):
        # q: [batch_size, seq_len, dim]
        # k: [batch_size, seq_len, dim]
        if v is None: v = torch.clone(k)
        batch_size, q_len, k_len, v_len = q.shape[0], q.shape[1], k.shape[1], v.shape[1]
        
        # 分离出多头，并将多头与batch_size放一起
        qx = self.w_q(q).view(batch_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).view(-1, q_len, self.hidden_dim) 
        kx = self.w_k(k).view(batch_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).view(-1, k_len, self.hidden_dim)
        vx = self.w_v(v).view(batch_size, v_len, self.n_head, self.hidden_dim)
        vx = vx.permute(2, 0, 1, 3).view(-1, v_len, self.hidden_dim)

        # 计算向量之间关系
        if self.method == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=1).expand(-1, -1, k_len, -1)
            qkx = torch.cat((qxx, kxx), dim=-1) # [bz*head, q_len, k_len, dim]
            score = torch.tanh(torch.matmul(qkx, self.weight))
        if self.method == 'self':
            score = torch.bmm(qx, kx.permute(0, 2, 1))
        if self.method == 'linear':
            qw = torch.matmul(qx, self.weight)
            score = torch.bmm(qw, kx.permute(0, 2, 1))

        # 对score处理，输出output
        if mask is not None: score = torch.add(score, (mask.reshape(score.shape)-1)*1e16)
        score = torch.softmax(score, dim=-1)
        out_head = torch.bmm(score, vx)
        output = torch.cat(torch.split(out_head, batch_size, dim=0), dim=-1)
        output = self.dropout(self.dense(output))

        return output, score


