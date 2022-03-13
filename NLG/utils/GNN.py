import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, alpha=0.2, concat=False):
        super(GATLayer, self).__init__()
        self.in_feats  = in_feats   # input dim
        self.out_feats = out_feats  # output dim
        self.drop_rate = dropout    # dropout rate for attention
        self.alpha     = alpha      # LeakyRelu coefficient
        self.concat    = concat     # method of concat or average of GAT
        self.leakyrelu = nn.LeakyReLU(self.alpha)  # LeakyRelu activate 

        self.W = nn.Linear(in_feats, out_feats)    # linear for initial nodes
        self.a = nn.Linear(2*out_feats, 1)         # linear for obtain weight

    def forward(self, inputs, adjs, masks=None):

        # 0. linear for input node: in_feats -> in_feats
        # [batch_size, node_num, hidden_dim], batch_size, nodes_num
        nodes, bs, N = self.W(inputs), inputs.size()[0], inputs.size()[1]
        
        # 1. calculate weights of each two nodes pair
        # 1.1. full assemble for nodes pair: [batch_size, N*N, hidden_dim]*2
        nodes_cat = [nodes.repeat(1,1,N).view(bs, N*N, -1), nodes.repeat(1,N,1)]
        nodes_assemble = torch.cat(nodes_cat, dim=2).view(bs, N, -1, 2*self.out_feats)
        # 1.2. calculate weight of each pair
        weights = self.leakyrelu(self.a(nodes_assemble).squeeze(dim=-1))

        # 2. weights only for connect nodes
        mask = -1e32 * torch.ones_like(weights) 
        # 2.1. find the connection, mask non-connect nodes
        attn = torch.where(adjs > 0, weights, mask)
        # 2.2. normalization for the new weights
        attn = F.softmax(attn, dim=2)
        # 2.3. dropout the weights
        attn = F.dropout(attn, self.drop_rate, training=self.training) # 数值会增大

        # 3. update nodes representation according weights
        new_nodes = []
        for per_w, per_n in zip(attn, nodes): # fetch a node
            new_nodes.append(torch.matmul(per_w, per_n)) # 更新sample的n个节点表示
        new_nodes = torch.stack(new_nodes)

        # 4. method-concat: return activate output; method-average: return direct output
        if self.concat:
            return F.elu(new_nodes)
        else:
            return new_nodes

class GAT(nn.Module):
    """
    paramters: (nfeat: input dim) - (nhid: hidden dim) - (nclass: output dim)
    note: GAT contain nheads GATLayer and a output GATLayer, contain two concat methods.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.1, alpha=0.2, nheads=8, concat=True, nlayer=1):
        super(GAT, self).__init__()
        self.drop_rate   = dropout
        self.dropout     = nn.Dropout(dropout)
        # stack many GATLayer
        self.attentions  = [GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] 
        # method-concat: nheads*nhid -> nclass; method-average: nhid -> nclass
        if concat:
            self.out_att = GATLayer(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        else:
            self.out_att = GATLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, masks=None):
        """
        x:   nodes
        adj: adjacent matrix
        """
        nodes = x
        x = F.dropout(x, self.drop_rate, training=self.training)  # dropout for training stage
        x = torch.cat([att(x, adj, masks) for att in self.attentions], dim=-1) # concat -> nhead*nhid
        x = F.dropout(x, self.drop_rate, training=self.training)  # dropout for training stage
        x = F.elu(self.out_att(x, adj, masks))   
        if masks is not None: x = torch.mul(x, masks) # recover padding value

        # residual connection
        return x + nodes



class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(GCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.use_bias = bias
        if self.use_bias:
            self.B = nn.Parameter(torch.FloatTensor(out_feats))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W)
        if self.use_bias:
            nn.init.zeros_(self.B)
    
    def forward(self, inputs, adjs):
        support = torch.mm(inputs, self.W)
        output = torch.spmm(adjs, support)
        if self.use_bias:
            return output + self.B
        else:
            return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.1, nheads=8, concat=True):
        super(GCN, self).__init__()
        self.drop_rate = dropout
        self.gcn_hid   = [GCNLayer(nfeat, nhid, dropout=dropout, concat=True) for _ in range(nheads)] 
        GCNLayer(nfeat, nhid)
        
        self.gcn_out   = GCNLayer(nhid*nheads, nclass)
       
    
    def forward(self, X, adj):
        X = F.relu(self.gcn1(X, adj))
        X = self.gcn2(X, adj)
        
        return F.log_softmax(X, dim=1)