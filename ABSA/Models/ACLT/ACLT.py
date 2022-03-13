

import torch, math
import torch.nn as nn
from transformers import BertModel


class ACLT(nn.Module):

    def __init__(self, args, Data):
        super(ACLT, self).__init__()
        self.args = args

    def forward(self, inputs):
        pass
