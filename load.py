# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import logging
import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, IANCNN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT,AOACNN
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from sklearn.utils import class_weight
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
import AttViz.text_attention as av

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        # if 'bert' in opt.model_name:
        #    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        #    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        #    self.model = opt.model_class(bert, opt).to(opt.device)
        # else:
        tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        embedding_matrix = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        self.Tok=tokenizer.word2idx
        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(
                self.trainset, [len(self.trainset)-valset_len, valset_len])
            #self.trainset, self.testset = random_split(
            #    self.trainset, [len(self.trainset)-valset_len, valset_len])
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(
                torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(
            n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device)
                          for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1)
                              == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(
                        train_loss, train_acc))
            # print(val_data_loader)
            val_acc, val_f1, val_precision, val_recall,val_f1_macro,val_precision_macro,val_recall_macro = self._evaluate_acc_f1(
                val_data_loader)
            logger.info(
                '> val_acc: {:.4f}, val_f1: {:.4f}, val_precision: {:.4f},val_recall: {:.4f},val_f1_macro: {:.4f},val_precision_macro: {:.4f},val_recall_macro: {:.4f}'.format(val_acc, val_f1, val_precision, val_recall,val_f1_macro,val_precision_macro,val_recall_macro))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(
                    self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
            #    break

        return path
    def AttentionViz(self,AttW,sent):
        cpt=0
        for i in AttW[0]:
            if(i>0):
                cpt+=1
                print(i)
        print(cpt,'------------')
        print((sent[0][0]).shape)
        words=[]
        idx2word = {}
        for word,i in self.Tok.items():
            idx2word[i] = word
        for i in sent[0][0]:
            if(int(i) != 0):
                words.append(idx2word[int(i)])
        word_num = len(words)
        #attention = AttW[0].tolist()
        attention = [(x+1.)/word_num*100 for x in range(word_num)]
        #import random
        #random.seed(42)
        #random.shuffle(attention)
        color = 'red'
        av.generate(words, attention, "./AttViz/sample.tex", color)
    def _evaluate_acc_f1(self, data_loader,test_mod=False):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all,t_inputs_all, weights_all = None, None, None,None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device)
                            for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs,context_final = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1)
                              == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                    weights_all = context_final
                    t_inputs_all = torch.cat(t_inputs,dim=0)
                    
                else:
                    t_targets_all = torch.cat(
                        (t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat(
                        (t_outputs_all, t_outputs), dim=0)
                    weights_all = torch.cat((weights_all, context_final), dim=0)
                    t_inputs_all =torch.cat((t_inputs_all, torch.cat(t_inputs,dim=0)), dim=0)
        if(test_mod==True):
                    #self.AttentionViz(xx,t_inputs)
            logger.info('{}'.format((torch.argmax(t_outputs_all, -1)).tolist()))
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(
            t_outputs_all, -1).cpu(), labels=[0, 1, 2, 3], average='weighted')
        precision = metrics.precision_score(t_targets_all.cpu(), torch.argmax(
            t_outputs_all, -1).cpu(), average='weighted')
        recall = metrics.recall_score(t_targets_all.cpu(), torch.argmax(
            t_outputs_all, -1).cpu(), average='weighted')
        f1_macro = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2, 3], average='macro')
        precision_macro = metrics.precision_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')
        recall_macro = metrics.recall_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')
        return acc, f1, precision, recall,f1_macro,precision_macro,recall_macro,weights_all,t_inputs_all

    def C_weight(self, data_loader):
        t_targets_all = None
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_targets = t_batch['polarity'].to(self.opt.device)
                if t_targets_all is None:
                    t_targets_all = t_targets
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
        t_targets_all=t_targets_all.tolist()
        class_weights = class_weight.compute_class_weight('balanced',
                                                 classes = numpy.unique(t_targets_all),
                                                 y= t_targets_all)       
        return class_weights
    def run(self):
        #train_data_loader = DataLoader(
        #    dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        # Loss and Optimizer
        #class_weights = self.C_weight(train_data_loader)
        #print('class_weights :',class_weights)
        #logger.info('class_weights :{}'.format(class_weights))
        #weights= torch.tensor(class_weights,dtype=torch.float).to(self.opt.device)
        #criterion = nn.CrossEntropyLoss(weight=weights)
        #_params = filter(lambda p: p.requires_grad, self.model.parameters())
        #optimizer = self.opt.optimizer(
        #    _params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        
        test_data_loader = DataLoader(
            dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        #val_data_loader = DataLoader(
        #    dataset=self.valset, batch_size=self.opt.batch_size, shuffle=True)

        #self._reset_params()
        #best_model_path = self._train(
        #    criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load('state_dict/ian_twitter_val_acc_0.8617'),strict=False)
        #print("Model's state_dict:")
        #for param_tensor in self.model.state_dict():
        #    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        
        test_acc, test_f1,precision,recall,test_f1_macro,test_precision_macro,test_recall_macro,weights_all,t_inputs_all = self._evaluate_acc_f1(test_data_loader,test_mod=True)
        torch.save(weights_all, 'weights.pt')
        torch.save(t_inputs_all, 'inputs.pt')
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, test_f1_macro: {:.4f}, test_precision_macro: {:.4f}, test_recall_macro: {:.4f}'.format(
            test_acc, test_f1, precision,recall,test_f1_macro,test_precision_macro,test_recall_macro))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='laptop',
                        type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=100, type=int,
                        help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name',
                        default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=300, type=int)
    parser.add_argument('--polarities_dim', default=4, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int,
                        help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.15, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm',
                        type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'ianCNN': IANCNN,
		'aoaCNN': AOACNN,
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
        'ianCNN': ['text_indices', 'cnn_aspect_indices'],
		'aoaCNN': ['text_indices', 'cnn_aspect_indices'],
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
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name,
                                     opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
