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
class HighwayCNN(nn.Module):

    def __init__(self, args):
        super(HighwayCNN, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        if len(Ks) > 1:
            print("current demo can not solve multiple kernel_sizes bug. "
                  "please modify the length of kernel_sizes is 1")
            exit()
        KK = []
        for K in Ks:
            KK.append(K + 1 if K % 2 == 0 else K)
        self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=D, kernel_size=(K, D), stride=(1, 1),
                                 padding=(K // 2, 0), dilation=1, bias=True) for K in Ks]
        if self.args.cuda is True:
            for conv in self.convs1:
                conv.cuda()
        in_feas = len(Ks) * Co
        self.fc1 = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)
        # Highway gate layer  T in the Highway formula
        self.gate_layer = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)

        if self.args.cuda is True:
            self.fc1.cuda()
            self.gate_layer.cuda()

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.args.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        in_fea = x.size(0)
        out_fea = x.size(1)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        x = torch.cat(x, 1)
        # normal layer in the formula is H
        out_fea = x.size(2)
        self.fc1 = self.init_Linear(in_fea=out_fea, out_fea=out_fea, bias=True)
        self.gate_layer = self.init_Linear(in_fea=out_fea, out_fea=out_fea, bias=True)
        list = []
        for i in range(x.size(0)):
            normal_fc = F.tanh(self.fc1(x[i]))
            # transformation gate layer in the formula is T
            transformation_layer = F.sigmoid(self.gate_layer(x[i]))
            # carry gate layer in the formula is C
            carry_layer = 1 - transformation_layer
            # formula Y = H * T + x * C
            allow_transformation = torch.mul(normal_fc, transformation_layer)
            allow_carry = torch.mul(x[i], carry_layer)
            information_flow = torch.add(allow_transformation, allow_carry)
            # follow for the next input
            information_flow = information_flow.unsqueeze(0)
            list.append(information_flow)
        information_flow = torch.cat(list, 0)
        information_flow = torch.transpose(information_flow, 1, 2)
        return information_flow


# HighWay recurrent model
class HighWayCNN_model(nn.Module):

    def __init__(self, args):
        super(HighWayCNN_model, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        self.embed = nn.Embedding(V, D)
        if args.word_Embedding is True:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # multiple HighWay layers List
        self.highway = nn.ModuleList([HighwayCNN(args) for _ in range(args.layer_num_highway)])
        self.output_layer = self.init_Linear(in_fea=self.args.embed_dim, out_fea=self.C, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.args.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        x = self.embed(x)
        # print(x.size())
        # self.output_layer = self.init_Linear(in_fea=x.size(2), out_fea=self.C, bias=True)
        for current_layer in self.highway:
            x = current_layer(x)
        x = torch.transpose(x, 1, 2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # print(x.size())
        output_layer = self.output_layer(x)
        return output_layer








