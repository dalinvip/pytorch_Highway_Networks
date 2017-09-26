# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import torch.nn.init as init
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)

# HighWay Networks model
class Highway(nn.Module):

    def __init__(self, args):
        super(Highway, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num

        in_feas = 1000
        self.fc1 = self.init_Linear(in_fea=in_feas, out_fea=D, bias=True)
        # Highway gate layer  T in the Highway formula
        self.gate_layer = self.init_Linear(in_fea=in_feas, out_fea=D, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.args.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        in_fea = x.size(0)
        out_fea = x.size(1)
        # init Linear for the same size
        self.fc1 = self.init_Linear(in_fea=out_fea, out_fea=out_fea, bias=True)
        self.gate_layer = self.init_Linear(in_fea=out_fea, out_fea=out_fea, bias=True)
        # normal layer in the formula is H
        normal_fc = F.tanh(self.fc1(x))
        # transformation gate layer in the formula is T
        transformation_layer = F.sigmoid(self.gate_layer(x))
        # carry gate layer in the formula is C
        carry_layer = 1 - transformation_layer
        # formula Y = H * T + x * C
        allow_transformation = torch.mul(normal_fc, transformation_layer)
        allow_carry = torch.mul(x, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)
        return information_flow


# HighWay recurrent model
class HighWay_model(nn.Module):

    def __init__(self, args):
        super(HighWay_model, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        self.embed = nn.Embedding(V, D)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        if args.word_Embedding is True:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # multiple HighWay layers List
        self.highway = nn.ModuleList([Highway(args) for _ in range(args.layer_num_highway)])
        self.output_layer = self.init_Linear(in_fea=D, out_fea=self.C, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.args.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        x = F.max_pool1d(x.permute(0, 2, 1), x.size(1)).squeeze(2)
        # self.output_layer = self.init_Linear(in_fea=x.size(1), out_fea=self.C, bias=True)
        for current_layer in self.highway:
            x = current_layer(x)
        output_layer = self.output_layer(x)
        if self.args.cuda is True:
            return output_layer.cuda()
        else:
            return output_layer








