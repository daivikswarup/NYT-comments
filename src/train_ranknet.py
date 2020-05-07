from data import dataset_ranking, todense
from ranknet_model import ranker
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import tqdm
import numpy as np
import pickle
import os
import sys


P = 5
vectype = sys.argv[1]
savefile = sys.argv[2]

def dcg(args, rels):
    score = 0.0
    for i in range(min(len(args), P)):
        score += (np.exp2(rels[args[i]]) -1.0)/np.log(2+i)
    return score


def compute_ndcg(preds, target):
    args = np.argsort(-preds.squeeze(1))
    rels = target/np.max(target)
    idcg = dcg(np.arange(len(args)), rels)
    dcg_ = dcg(args, rels)
    return dcg_ / idcg




def eval(model, dataset):
    model.eval()
    ndcgs = []
    for vecs, scores in dataset.eval_batches():
        vecs = torch.tensor(vecs).double().cuda()
        preds = model(vecs)
        ndcgs.append(compute_ndcg(preds.cpu().detach().numpy(), scores))
    return np.mean(ndcgs), ndcgs

def eval_random(dataset):
    ndcgs = []
    for vecs, scores in dataset.eval_batches():
        # vecs = torch.tensor(vecs).cuda()
        # preds = model(vecs)
        preds = np.random.randn(vecs.shape[0], 1)
        # preds = np.zeros((vecs.shape[0],1))
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
    best_ndcg = -1
    for e in tqdm.trange(n_epochs):
        model.train()
        for x1, x2, targ in train_dataset.get_batches(bs):
            optimizer.zero_grad()
            x1, x2, targ = torch.tensor(x1).double().cuda(),\
                               torch.tensor(x2).double().cuda(),\
                                    torch.tensor(targ).cuda()
            pred = model(x1, x2)
            loss = loss_func(pred.squeeze(-1), targ)
            loss.backward()
            optimizer.step()
        ndcg, all_ndcg = eval(model, val_dataset) 
        if ndcg > best_ndcg:
            torch.save(model.state_dict(), savefile)
            best_ndcg = ndcg
            with open(savefile+'.preds.pkl', 'wb') as f:
                pickle.dump(all_ndcg, f)
        print('NDCG after %d epochs = %f'%(e, ndcg))
        torch.save(model.state_dict(), 'ranker.pt')
    return model

if __name__ == '__main__':
    if os.path.exists('dataset_%s.pkl'%vectype):
        with open('dataset_%s.pkl'%vectype, 'rb') as f:
            traindata, valdata = pickle.load(f)
    else:
        traindata = dataset_ranking('../data/train', vectype=vectype)
        vectorizer = traindata.vectorizer
        valdata = dataset_ranking('../data/val', vectorizer, vectype=vectype)
        with open('dataset_%s.pkl'%vectype, 'wb') as f:
            pickle.dump((traindata, valdata), f)
    print(traindata.num_features)
    model = ranker(traindata.num_features).double().cuda()
    if os.path.exists(savefile):
        model.load_state_dict(torch.load(savefile))
    else:
        # print(eval(model, valdata))
        train(model, traindata, valdata, 20, 3e-4, 256)
    
    # print(eval(model, valdata)[0])
    # print(eval_random(valdata))


    # Evaluate on the test article
    if len(sys.argv) > 3:
        fname = sys.argv[3]
        fdata = valdata.vectorized_fmap[fname]
        vectors = [com[1] for com in fdata]
        stacked = np.stack(todense(vectors))
        scores = model(torch.tensor(stacked).double().cuda())
        print(scores.cpu().detach())

