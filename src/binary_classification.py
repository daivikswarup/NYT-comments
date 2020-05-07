# simple baseline
from data import dataset
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
from sklearn import preprocessing
import pickle
from sklearn.neural_network import MLPClassifier

vectype = sys.argv[1]
outfile = sys.argv[2]

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


with open(outfile, 'wb') as f:
    pickle.dump(clf.predict(valx), f)

with open('gt.pkl', 'wb') as f:
    pickle.dump(valy, f)


with open('clf.pkl', 'wb') as f:
    pickle.dump([clf, vectorizer], f)

if len(sys.argv) > 3:
    with open(sys.argv[3], 'r') as f:
        lines = f.readlines()
    comments = [l.split('\t')[0] for l in lines]
    vecs = vectorizer.transform(comments)
    preds = clf.predict(vecs)
    with open('predictions_1art.pkl', 'wb') as f:
        pickle.dump(preds, f)
