from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from models import MFGCN
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data.sampler import *
# AMGCN
###################
#(AS)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, default='uai')
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default='20')
    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)
    cuda = not config.no_cuda and torch.cuda.is_available()
    theta =0.9
    beta =1
    ep =config.epochs/2
    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test, idx = load_data(config)
    sadj = sadj.cuda()
    fadj = fadj.cuda()
    model = MFGCN(nfeat=config.fdim,
                  nhid1=config.nhid1,
                  nhid2=config.nhid2,
                  nclass=config.class_num,
                  n=config.n,
                  dropout=config.dropout,
                  sadj=sadj,
                  fadj=fadj,
                  )
    if cuda:
        model.cuda()
        features = features.cuda()

        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    for name, param in model.named_parameters():  # 查看可优化的参数有哪些
        if param.requires_grad:
            print(name)



    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def set_label(sim_value,labels,sim):
         index_label = []
         label =[]
         a=0
         b=0
         for i in range(int(len(idx))):
             if sim[idx[i],sim_value[idx[i]]]>theta:
                 if(labels[idx[i]]!=labels[idx_train[sim_value[idx[i]]]]):
                     # print('wrong',i, idx[i], idx_train[sim_value[idx[i]]], labels[idx[i]],
                     #       sim[idx[i], sim_value[idx[i]]],labels[idx_train[sim_value[idx[i]]]])
                     a= a+1
                 if(labels[idx[i]]==labels[idx_train[sim_value[idx[i]]]]):
                     # print('Right',i, idx[i], idx_train[sim_value[idx[i]]], labels[idx[i]],
                     #       sim[idx[i], sim_value[idx[i]]],labels[idx_train[sim_value[idx[i]]]])
                     b = b+1
                 index_label.append(idx[i])
                 label.append(labels[idx_train[sim_value[idx[i]]]])
         print('a',a,'b',b)
         return  index_label,label
    def rand(index_label,label):
        s =int(len(index_label)*0.8)
        ran_idx = []
        ran_label = []
        t = np.arange(len(index_label))
        for i in RandomSampler(t, replacement=True, num_samples=s):
            ran_idx.append(index_label[i])
            ran_label.append(label[i])
        return ran_idx,ran_label

    def train(model, epochs):
        model.train()
        optimizer.zero_grad()

        output, s, emb1, com1, com2, emb2, emb, estimated_adj, pslb, cluster_assignment1, weight_label = model(
            features, sadj, fadj)

        sim = cosine_similarity(pslb.cpu().detach().numpy() )

        sim = sim - np.diag(np.diagonal(sim))
        sim = torch.from_numpy(sim)
        sim = sim[:,idx_train]
        sim_value = sim.argmax(dim=1)
        index_label, label_add = set_label(sim_value,labels,sim)
        label_add=torch.tensor(label_add).cuda()
        print('label_add.shape',label_add.shape)
        loss_class = F.nll_loss(output[idx_train], labels[idx_train])
        loss_l1 = torch.norm(estimated_adj, 1)
        loss_fro1 = torch.norm(estimated_adj - sadj, p='fro')
        loss_fro2 = torch.norm(estimated_adj - fadj, p='fro')
        loss_l2 = torch.norm(estimated_adj, p='nuc')
        if epochs>int(ep):
         a,b = rand(index_label,label_add)
         a=torch.tensor(a).clone().detach()
         b=torch.tensor(b).cuda()
         loss_add = F.nll_loss(output[a],b)
         loss = loss_class + loss_add*beta+(loss_l1+loss_l2+loss_fro2+loss_fro1)*0.1
        else:
            loss = loss_class +(loss_l1+loss_l2+loss_fro2+loss_fro1)*0.1

        acc = accuracy(output[idx_train], labels[idx_train])

        loss.backward()
        optimizer.step()
        acc_test, macro_f1, emb_test = main_test(model)
        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.item()),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()))
        return loss.item(), acc_test.item(), macro_f1.item(), emb_test


    def main_test(model):
        model.eval()
        output, att, emb1, com1, com2, emb2, emb, estimator, pslb, cluster_assignment1, weight_label = model(features,
                                                                                                             sadj, fadj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb


    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(config.epochs):
        loss, acc_test, macro_f1, emb = train(model, epoch)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print(args.dataset, args.labelrate,beta,theta)
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))
