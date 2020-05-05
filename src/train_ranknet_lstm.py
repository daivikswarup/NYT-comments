from data import dataset_ranking_lstm
from ranknet_model import LSTM_ranker
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import tqdm
import numpy as np
import pickle
import os
from settings import *

def dcg(args, rels):
    score = 0.0
    for i in range(len(args)):
        score += (np.exp2(rels[args[i]]) -1.0)/np.log(2+i)
    return score


def compute_ndcg(preds, target):
    args = np.argsort(-preds)
    rels = target/np.max(target)
    idcg = dcg(np.arange(len(args)), rels)
    dcg_ = dcg(args, rels)
    return dcg_ / idcg




def eval(model, dataset):
    model.eval()
    ndcgs = []
    for vecs, l , scores in dataset.eval_batches():
        num_comments = len(vecs)
        pred_batches = []
        for i in range(0, num_comments, BATCH_SIZE):
            pred_batches.append(model(vecs[i:i+BATCH_SIZE],l[i:i+BATCH_SIZE]))

        preds = torch.cat(pred_batches)
        ndcgs.append(compute_ndcg(preds.cpu().detach().numpy(), scores))
    return np.mean(ndcgs)

def eval_random(dataset):
    ndcgs = []
    for vecs,l, scores in dataset.eval_batches():
        # vecs = torch.tensor(vecs).cuda()
        # preds = model(vecs)

        # preds = np.random.randn(vecs.shape[0])
        # preds = np.zeros(vecs.shape[0])
        preds = l.detach().cpu().numpy()
        ndcgs.append(compute_ndcg(preds, scores))
    return np.mean(ndcgs)

        

# def loss_func(pred, target):
#     p_plus = F.logsigmoid(pred)
#     p_minus = F.logsigmoid(-pred)
#     loss = torch.sum(-target * p_plus -(1-target)*p_minus)
#     return loss


def train(model, train_dataset,val_dataset, n_epochs, lr, bs): 
    optimizer = optim.Adam(model.parameters(), lr= lr)
    loss_func = nn.BCEWithLogitsLoss() 
    for e in tqdm.trange(n_epochs):
        model.train()
        for x1, l1, x2, l2, targ in train_dataset.get_batches(bs):
            optimizer.zero_grad()
            pred = model(x1, l1, x2, l2)
            loss = loss_func(pred.squeeze(-1), targ)
            loss.backward()
            optimizer.step()
        ndcg = eval(model, val_dataset) 
        print('NDCG after %d epochs = %f'%(e, ndcg))
        torch.save(model.state_dict(), 'ranker_lstm.pt')
    return model

if __name__ == '__main__':
    if os.path.exists('dataset_lstm.pkl'):
        with open('dataset_lstm.pkl', 'rb') as f:
            traindata, valdata = pickle.load(f)
    else:
        #traindata = dataset_ranking_lstm('../data/train', vocab_size=VOCAB_SIZE)
        # vocab  = traindata.vocab
        vocab = None
        valdata = dataset_ranking_lstm('../data/val', vocab)
        # with open('dataset_lstm.pkl', 'wb') as f:
            # pickle.dump((traindata, valdata), f)
        #     pass
    print(eval_random(valdata))
    exit(0)
    model = LSTM_ranker(traindata.vocab.vocab_size,\
                        EMBEDDING_DIM,NUM_LAYERS,traindata.vocab.pad_id).double().cuda()
    if os.path.exists('ranker_lstm.pt'):
        model.load_state_dict(torch.load('ranker_lstm.pt'))
    else:
        print(eval(model, valdata))
        train(model, traindata, valdata, 10, 3e-4, 256)
    print(eval(model, valdata))
    print(eval_random(valdata))

