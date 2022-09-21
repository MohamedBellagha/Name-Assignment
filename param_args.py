# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import argparse
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC

import torch
import torch.nn as nn
def opt_():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', default='bert_spc', type=str)
	parser.add_argument('--dataset', default='laptop',
						type=str, help='twitter, restaurant, laptop')
	parser.add_argument('--optimizer', default='adam', type=str)
	parser.add_argument('--initializer', default='xavier_uniform_', type=str)
	parser.add_argument('--lr', default=2e-5, type=float,
						help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
	parser.add_argument('--dropout', default=0.1, type=float)
	parser.add_argument('--l2reg', default=0.01, type=float)
	parser.add_argument('--num_epoch', default=10, type=int,
						help='try larger number for non-BERT models')
	parser.add_argument('--batch_size', default=16, type=int,
						help='try 16, 32, 64 for BERT models')
	parser.add_argument('--log_step', default=10, type=int)
	parser.add_argument('--embed_dim', default=300, type=int)
	parser.add_argument('--hidden_dim', default=300, type=int)
	parser.add_argument('--bert_dim', default=768, type=int)
	parser.add_argument('--pretrained_bert_name',
						default='bert-base-uncased', type=str)
	parser.add_argument('--max_seq_len', default=150, type=int)
	parser.add_argument('--polarities_dim', default=3, type=int)
	parser.add_argument('--hops', default=3, type=int)
	parser.add_argument('--patience', default=5, type=int)
	parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
	parser.add_argument('--seed', default=1234, type=int,
						help='set seed for reproducibility')
	parser.add_argument('--valset_ratio', default=0, type=float,
						help='set ratio between 0 and 1 for validation support')
	# The following parameters are only valid for the lcf-bert model
	parser.add_argument('--local_context_focus', default='cdm',
						type=str, help='local context focus mode, cdw or cdm')
	parser.add_argument('--SRD', default=3, type=int,
						help='semantic-relative-distance, see the paper of LCF-BERT model')

	model_classes = {
		'lstm': LSTM,
		'td_lstm': TD_LSTM,
		'tc_lstm': TC_LSTM,
		'atae_lstm': ATAE_LSTM,
		'ian': IAN,
		'memnet': MemNet,
		'ram': RAM,
		'cabasc': Cabasc,
		'tnet_lf': TNet_LF,
		'aoa': AOA,
		'mgan': MGAN,
		'asgcn': ASGCN,
		'bert_spc': BERT_SPC,
		'aen_bert': AEN_BERT,
		'lcf_bert': LCF_BERT,
		# default hyper-parameters for LCF-BERT model is as follws:
		# lr: 2e-5
		# l2: 1e-5
		# batch size: 16
		# num epochs: 5
	}
	dataset_files = {
		'twitter': {
			'train': './datasets/acl-14-short-data/train.raw',
			'test': './datasets/acl-14-short-data/test.raw'
		},
		'restaurant': {
			'train': './datasets/semeval14/Restaurants_Train.xml.seg',
			'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
		},
		'laptop': {
			'train': './datasets/semeval14/Laptops_Train.xml.seg',
			'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
		}
	}
	input_colses = {
		'lstm': ['text_indices'],
		'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
		'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
		'atae_lstm': ['text_indices', 'aspect_indices'],
		'ian': ['text_indices', 'aspect_indices'],
		'memnet': ['context_indices', 'aspect_indices'],
		'ram': ['text_indices', 'aspect_indices', 'left_indices'],
		'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
		'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
		'aoa': ['text_indices', 'aspect_indices'],
		'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
		'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
		'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
		'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
		'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
	}
	initializers = {
		'xavier_uniform_': torch.nn.init.xavier_uniform_,
		'xavier_normal_': torch.nn.init.xavier_normal_,
		'orthogonal_': torch.nn.init.orthogonal_,
	}
	optimizers = {
		'adadelta': torch.optim.Adadelta,  # default lr=1.0
		'adagrad': torch.optim.Adagrad,  # default lr=0.01
		'adam': torch.optim.Adam,  # default lr=0.001
		'adamax': torch.optim.Adamax,  # default lr=0.002
		'asgd': torch.optim.ASGD,  # default lr=0.01
		'rmsprop': torch.optim.RMSprop,  # default lr=0.01
		'sgd': torch.optim.SGD,
	}

	#ins = Instructor(opt)
	#ins.run()
	opt = parser.parse_args()
	return opt
	 