from data import dataset_lstm
from lstm_model import lstm
import torch.nn as nn
import torch
import numpy as np
from settings import *


traindata = dataset_lstm('train.txt', vocab_size = VOCAB_SIZE)
vocab = traindata.vocab
pad_id = traindata.pad_id
valdata = dataset_lstm('val.txt', vocab = vocab)

model = lstm(VOCAB_SIZE, EMBEDDING_DIM, NUM_LAYERS, pad_id)
model = model.to(device=DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def eval(model, dataset):
    model.eval()
    predictions = []
    targets = []
    for x, l, y in dataset.get_batches(batch_size=BATCH_SIZE):
        p = model(x, l).cpu().detach().numpy()
        targets.append(y.cpu().detach().numpy())
        predictions.append(p)
    predictions = np.concatenate(predictions) >= 0.5
    targets = np.concatenate(targets)
    return np.mean(predictions==targets)
        
def train(model, train_dataset, val_dataset):
    for epoch in range(NUM_EPOCHS):
        model.train()
        for x, l, y in train_dataset.get_batches(batch_size=BATCH_SIZE):
            optimizer.zero_grad()
            p = model(x,l)
            loss = criterion(p,y)
            loss.backward()
            optimizer.step()
        accuracy = eval(model, val_dataset)
        print('validation accuracy = %f'%accuracy)


train(model, traindata, valdata)
