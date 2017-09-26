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


class BiLSTMList_Cat(nn.Module):

    def __init__(self, args):
        super(BiLSTMList_Cat, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, bias=True, bidirectional=True, dropout=args.dropout)
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
        # print(x.size())
        x, hidden = self.bilstm(x, hidden)
        x = torch.transpose(x, 0, 1)

        # the way to convert 3D tensor to Linear
        x = x.contiguous()
        information_flow = x.view(x.size(0) * x.size(1), x.size(2))
        information_flow = self.convert_layer(information_flow.permute(0, 1))
        x = information_flow.view(x.size(0), x.size(1), information_flow.size(1))
        # print(x.size())

        '''
        list = []
        for i in range(x.size(0)):
            information_flow = x[i]
            information_flow = self.convert_layer(information_flow)
            information_flow = information_flow.unsqueeze(0)
            list.append(information_flow)
        x = torch.cat(list, 0)
        '''
        x = torch.transpose(x, 0, 1)
        return x, hidden


# HighWay recurrent model
class BiLSTMList_model_CAT(nn.Module):

    def __init__(self, args):
        super(BiLSTMList_model_CAT, self).__init__()
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
        self.highway = nn.ModuleList([BiLSTMList_Cat(args) for _ in range(args.layer_num_highway)])
        # self.output_layer = self.init_Linear(in_fea=self.args.embed_dim, out_fea=self.C, bias=True)
        self.output_layer = self.init_Linear(in_fea=self.args.embed_dim, out_fea=self.C, bias=True)
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.args.cuda is True:
            return linear.cuda()
        else:
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
        self.hidden = self.init_hidden(self.args.lstm_num_layers, x.size(1))
        for current_layer in self.highway:
            x, self.hidden = current_layer(x, self.hidden)
            if self.args.layer_num_highway == 1:
                cat_x = x
            else:
                if current_layer == self.highway[0]:
                    cat_x = x
                else:
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
        # print()
        if self.args.cuda is True:
            return output_layer.cuda()
        else:
            return output_layer








