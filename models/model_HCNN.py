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
        # if len(Ks) > 1:
        #     print("current demo can not solve multiple kernel_sizes bug. "
        #           "please modify the length of kernel_sizes is 1")
        #     exit()
        KK = []
        for K in Ks:
            KK.append(K + 1 if K % 2 == 0 else K)
        print(KK)
        self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=D, kernel_size=(K, D), stride=(1, 1),
                                 padding=(K // 2, 0), dilation=1, bias=True) for K in KK]
        if args.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_uniform(conv.weight.data, gain=np.sqrt(args.init_weight_value))
        if self.args.cuda is True:
            for conv in self.convs1:
                conv.cuda()
        in_feas = len(Ks) * Co
        self.fc1 = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)
        # Highway gate layer  T in the Highway formula
        self.gate_layer = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.args.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        source_x = x
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        x = torch.cat(x, 1)
        # in the formula is the H
        normal_fc = torch.transpose(x, 1, 2)
        self.fc1 = self.init_Linear(in_fea=self.args.embed_dim * len(self.args.kernel_sizes), out_fea=self.args.embed_dim, bias=True)
        self.gate_layer = self.init_Linear(in_fea=self.args.embed_dim, out_fea=self.args.embed_dim *
                                                                                len(self.args.kernel_sizes), bias=True)

        source_x = source_x.contiguous()
        information_source = source_x.view(source_x.size(0) * source_x.size(1), source_x.size(2))
        information_source = self.gate_layer(information_source)
        information_source = information_source.view(source_x.size(0), source_x.size(1), information_source.size(1))
        information_source = torch.transpose(information_source, 1, 2)

        # the formula is Y = H * T + x * C
        # transformation gate layer in the formula is T
        transformation_layer = F.sigmoid(information_source)
        carry_layer = 1 - transformation_layer
        allow_transformation = torch.mul(normal_fc, transformation_layer)
        allow_carry = torch.mul(information_source, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)

        information_flow = information_flow.contiguous()
        information_convert = information_flow.view(information_flow.size(0) * information_flow.size(1), information_flow.size(2))
        information_convert = self.fc1(information_convert)
        information_convert = information_convert.view(information_flow.size(0), information_flow.size(1), information_convert.size(1))
        return information_convert


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
        if self.args.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        x = self.embed(x)
        for current_layer in self.highway:
            x = current_layer(x)
        x = torch.transpose(x, 1, 2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = F.relu(x)
        output_layer = self.output_layer(x)
        return output_layer








