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


class HBiLSTM_CAT(nn.Module):

    def __init__(self, args):
        super(HBiLSTM_CAT, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, bias=True, bidirectional=True
                              , dropout=args.dropout)
        in_feas = self.hidden_dim
        self.fc1 = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)
        # Highway gate layer  T in the Highway formula
        self.gate_layer = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)
        # if bidirection convert dim
        self.convert_layer = self.init_Linear(in_fea=self.args.lstm_hidden_dim * 2,
                                              out_fea=self.args.embed_dim, bias=True)
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.cuda is True:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)))

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.args.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x, hidden):
        # handle the source input x
        source_x = x
        # print(x)
        # print(source_x)
        x, hidden = self.bilstm(x, hidden)
        normal_fc = torch.transpose(x, 0, 1)
        x = torch.transpose(x, 0, 1)
        x = torch.transpose(x, 1, 2)
        # print(x.size())
        # normal layer in the formula is H
        in_fea = self.args.embed_dim
        out_fea = self.args.lstm_hidden_dim * 2
        self.fc1 = self.init_Linear(in_fea=in_fea, out_fea=out_fea, bias=True)
        self.gate_layer = self.init_Linear(in_fea=in_fea, out_fea=out_fea, bias=True)
        # print(self.fc1)
        # print(self.gate_layer)
        # print(self.convert_layer)
        list = []
        source_x = torch.transpose(source_x, 0, 1)
        # print(sourxe_x.size())
        for i in range(source_x.size(0)):
            # convert the dim im bidirection
            # information_flow = self.convert_layer(torch.transpose(information_flow, 0, 1))
            # print(source_x[i].size())
            information_source = self.gate_layer(source_x[i])
            # print(information_source.size())
            information_source = information_source.unsqueeze(0)
            list.append(information_source)
        information_source = torch.cat(list, 0)

        # transformation gate layer in the formula is T
        transformation_layer = F.sigmoid(information_source)
        # carry gate layer in the formula is C
        carry_layer = 1 - transformation_layer
        # formula Y = H * T + x * C
        allow_transformation = torch.mul(normal_fc, transformation_layer)

        # # you can choose the strategy that zero-padding
        # zeros = torch.zeros(source_x.size(0), source_x.size(1), carry_layer.size(2) - source_x.size(2))
        # if self.args.cuda is True:
        #     source_x = Variable(torch.cat((zeros, source_x.data), 2)).cuda()
        # else:
        #     source_x = Variable(torch.cat((zeros, source_x.data), 2))
        # allow_carry = torch.mul(source_x, carry_layer)

        # the information_source compare to the source_x is for the same size of x,y,H,T
        allow_carry = torch.mul(information_source, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)
        # print("wwwwwww", information_flow.size())
        # print(information_flow)

        convert = []
        for j in range(information_flow.size(0)):
            # print(information_flow[i].size())
            information_convert = self.convert_layer(information_flow[j])
            # print(information_convert.size())
            information_convert = information_convert.unsqueeze(0)
            convert.append(information_convert)
        information_convert = torch.cat(convert, 0)
        # print("information_convert ", information_convert.size())

        # information_flow = torch.transpose(information_flow, 1, 2)
        information_convert = torch.transpose(information_convert, 0, 1)
        # print(information_flow.size())
        # if self.args.cuda is True:
        #     return information_convert.cuda(), hidden.cuda()
        # else:
        return information_convert, hidden
        # return x, hidden


# HighWay recurrent model
class HBiLSTM_CAT_model(nn.Module):

    def __init__(self, args):
        super(HBiLSTM_CAT_model, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        self.embed = nn.Embedding(V, D)
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        if args.word_Embedding is True:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # multiple HighWay layers List
        self.highway = nn.ModuleList([HBiLSTM_CAT(args) for _ in range(args.layer_num_highway)])
        # self.output_layer = self.init_Linear(in_fea=self.args.embed_dim * self.args.layer_num_highway, out_fea=self.C, bias=True)
        self.output_layer = self.init_Linear(in_fea=self.args.embed_dim, out_fea=self.C, bias=True)
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        return linear

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.cuda is True:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)))


    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        # print(x.size())
        # self.output_layer = self.init_Linear(in_fea=self.args.lstm_hidden_dim * 2, out_fea=self.C, bias=True)
        self.hidden = self.init_hidden(self.num_layers, x.size(1))
        # print(self.hidden)
        for current_layer in self.highway:
            x, self.hidden = current_layer(x, self.hidden)
            if self.args.layer_num_highway == 1:
                cat_x = x
            else:
                if current_layer == self.highway[0]:
                    cat_x = x
                else:
                    # print(cat_x.size())
                    # two way cat
                    # cat_x = torch.cat((cat_x, x), 2)
                    cat_x = torch.cat((cat_x, x), 0)
                    # print(cat_x.size())

        x = cat_x
        x = torch.transpose(x, 0, 1)
        x = torch.transpose(x, 1, 2)
        x = F.tanh(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = F.tanh(x)
        output_layer = self.output_layer(x)
        # print(output_layer.size())
        if self.args.cuda is True:
            return output_layer.cuda()
        else:
            return output_layer








