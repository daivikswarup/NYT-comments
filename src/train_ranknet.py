from data import dataset_ranking
from ranknet_model import ranker
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import tqdm
import numpy as np

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
    for vecs, scores in dataset.eval_batches():
        vecs = torch.tensor(vecs)
        preds = model(vecs)
        ndcgs.append(compute_ndcg(preds.cpu().detach().numpy(), scores))
    return np.mean(ndcgs)
        




def train(model, train_dataset,val_dataset, n_epochs, lr, bs): 
    optimizer = optim.Adam(model.parameters(), lr= lr)
    for e in tqdm.trange(n_epochs):
        model.train()
        for x1, x2 in train_dataset.get_batches(bs):
            optimizer.zero_grad()
            x1, x2 = torch.tensor(x1), torch.tensor(x2)
            pred = model(x1, x2)
            loss = -torch.sum(pred)
            loss.backward()
            optimizer.step()
        ndcg = eval(model, val_dataset) 
        print('NDCG after %d epochs = %f'%(e, ndcg))
    torch.save(model.state_dict, 'ranker.pt')
    return model

if __name__ == '__main__':
    traindata = dataset_ranking('../data/train')
    vectorizer = traindata.vectorizer
    valdata = dataset_ranking('../data/val', vectorizer)
    model = ranker(traindata.num_features).double()
    train(model, traindata, valdata, 10, 3e-4, 512)

