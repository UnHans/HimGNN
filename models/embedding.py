from functools import partial
import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
#from layers import layer
from dgl.nn import NNConv
from .MLP import MLP
import torch.nn.functional as F
class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel)
        #self.bn=nn.LayerNorm(in_channel)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        size = x.size()
        x = x.view(-1, x.size()[-1], 1)
        x = self.bn(x)
        x = x.view(size)
        if self.act is not None:
            x = self.act(x)
        return x
