import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, kernel_size=None, step=3):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.conv      = nn.Conv1d(self.input_dim, self.input_dim, step, padding=0)
        self.pool      = nn.MaxPool1d(40)
        self.linear    = nn.Linear(100*3,1)
        self.relu      = nn.ReLU()

    def forward(self, inputs):
        padding  = torch.zeros(inputs.shape[0], 1, inputs.shape[2])
        inputs_p = torch.cat((padding, inputs, padding), dim=1)
        conv_out = self.conv(inputs_p.permute(0, 2, 1))

        maxpool  = nn.MaxPool1d(kernel_size=conv_out.shape[-1])
        pool_out = maxpool(conv_out)

        return pool_out.permute(0, 2, 1)