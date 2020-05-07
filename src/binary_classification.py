# simple baseline
from data import dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

vectype = sys.argv[1]

traindata = dataset('train_80_20.txt', vectype=vectype)
vectorizer = traindata.vectorizer
valdata = dataset('val_80_20.txt', vectorizer,vectype=vectype)
testdata = dataset('test_80_20.txt', vectorizer,vectype=vectype)

trainx, trainy = traindata.get_xy()
valx, valy = valdata.get_xy()
print(trainy)


print('training')
clf = LogisticRegression(random_state=0, verbose=1,max_iter=10000).fit(trainx, trainy)
# clf = MLPClassifier(random_state=0, verbose=1,max_iter=1000).fit(trainx, trainy)

print(clf.score(valx, valy))



