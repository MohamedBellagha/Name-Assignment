# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F


class IANCNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(IANCNN, self).__init__()
        filter_size = 7
        n_filters = 1#150
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        #self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        #self.conv = nn.Conv1d(in_channels=opt.embed_dim, out_channels = num_filters, kernel_size = filter_size, stride = 1)
        # self.conv=nn.Conv2d(in_channels = opt.embed_dim,
        #                    out_channels = num_filters, kernel_size = filter_size)
        # CNN
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, opt.embed_dim))
        #self.lstm_aspect = DynamicLSTM(n_filters, opt.hidden_dim, num_layers=1, batch_first=True)
        
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        #self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        
        self.lstm_aspect = nn.LSTM(n_filters, opt.hidden_dim,num_layers=1,bias=True, batch_first=True, dropout=0)
        
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        ################## CNN #####################
        # x = [sent len, batch size]
       #aspect_indices = aspect_indices.permute(1, 0)
        # x = [batch size, sent len]
        aspect = self.embed(aspect_indices)
        # embedded = [batch size, sent len, emb dim]
        embedded = aspect.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        # conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        # x = [sent len- filter_size, batch size, n_filters]

        #print(x.shape)
        #conv_len = torch.sum(x != 0, dim=-1)
        aspect, (_, _) = self.lstm_aspect(x)
        aspect=aspect.permute(1, 0, 2)
        aspect_len = aspect_len.detach().clone().to(self.opt.device)#torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))        
		
        text_raw_len = text_raw_len.detach().clone().to(self.opt.device)#torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(
            context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out
