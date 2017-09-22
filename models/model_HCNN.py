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


class HCNN(nn.Module):

    def __init__(self, args):
        super(HCNN, self).__init__()
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
        in_feas = len(Ks) * Co
        self.fc1 = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)
        # Highway gate layer  T in the Highway formula
        self.gate_layer = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        return linear

    def forward(self, x):
        source_x = x
        # print("source_x ", source_x.size())
        # print(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        x = torch.cat(x, 1)
        # in the formula is the H
        normal_fc = torch.transpose(x, 1, 2)
        out_fea = x.size(1)
        self.fc1 = self.init_Linear(in_fea=out_fea, out_fea=out_fea, bias=True)
        self.gate_layer = self.init_Linear(in_fea=out_fea, out_fea=out_fea, bias=True)
        # print(x.size())
        # print(self.fc1)
        # print(self.gate_layer)
        list = []
        # source_x = torch.transpose(source_x, 0, 1)
        # print(source_x.size())
        for i in range(source_x.size(0)):
            # print(source_x[i].size())
            information_source = self.gate_layer(source_x[i])
            # print(information_source.size())
            information_source = information_source.unsqueeze(0)
            list.append(information_source)
        information_source = torch.cat(list, 0)
        # print("wwww", information_source.size())
        # print(information_source)
        # the formula is Y = H * T + x * C
        # transformation gate layer in the formula is T
        transformation_layer = F.sigmoid(information_source)
        # print(transformation_layer.size())
        # carry gate layer in the formula is C
        carry_layer = 1 - transformation_layer
        # print(carry_layer.size())
        allow_transformation = torch.mul(normal_fc, transformation_layer)
        # print(sourxe_x.size())
        # allow_carry = torch.mul(information_source, carry_layer)
        allow_carry = torch.mul(source_x, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)
        # print(information_flow.size())
        # information_flow = torch.transpose(information_flow, 1, 2)
        return information_flow


# HighWay recurrent model
class HCNN_model(nn.Module):

    def __init__(self, args):
        super(HCNN_model, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        self.embed = nn.Embedding(V, D)
        if args.word_Embedding is True:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # multiple HighWay layers List
        self.highway = nn.ModuleList([HCNN(args) for _ in range(args.layer_num_highway)])
        self.output_layer = self.init_Linear(in_fea=self.args.embed_dim, out_fea=self.C, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        return linear

    def forward(self, x):
        x = self.embed(x)
        # print(x.size())
        # self.output_layer = self.init_Linear(in_fea=x.size(2), out_fea=self.C, bias=True)
        for current_layer in self.highway:
            # print(current_layer)
            x = current_layer(x)
        # print(x.size())
        x = torch.transpose(x, 1, 2)
        # print("wewew", x.size())
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        output_layer = self.output_layer(x)
        return output_layer








