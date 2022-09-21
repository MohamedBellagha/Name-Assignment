# -*- coding: utf-8 -*-
# file: aoa.py
# author: gene_zc <gene_zhangchen@163.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.CNNmodel import TextClassifier
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
##
class AOACNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AOACNN, self).__init__()
        n_filters = 150
        self.opt = opt
        filter_size = opt.filter_size
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        #self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = nn.LSTM(n_filters, opt.hidden_dim,num_layers=1,bias=True, batch_first=True,dropout=0,bidirectional=True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, opt.embed_dim))
        self.dense = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices = inputs[0] # batch_size x seq_len
        aspect_indices = inputs[1] # batch_size x seq_len
        ctx_len = torch.sum(text_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        ctx = self.embed(text_indices) # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices) # batch_size x seq_len x embed_dim
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) #  batch_size x (ctx) seq_len x 2*hidden_dim
        ###
        embedded = asp.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        # conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        asp_out, (_, _) = self.asp_lstm(x) # batch_size x (asp) seq_len x 2*hidden_dim

        asp_out=asp_out.permute(1, 0, 2)
        #print(ctx_out.size())
        #print(asp_out.size())
        interaction_mat = torch.matmul(ctx_out, torch.transpose(asp_out, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True) # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1) # batch_size x 2*hidden_dim
        out = self.dense(weighted_sum) # batch_size x polarity_dim

        return out