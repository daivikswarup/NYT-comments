# Evaluating Comments for Constructiveness  
This work was done as a part of CS685.  
Author: Daivik Swarup  

Download data from [here](https://www.kaggle.com/aashita/nyt-comments)  
## Preprocessing
Split data into train, test, val splits:
```bash
python preprocess.py
```

For classification, create thresholded text files:  
```bash
python preprocess_threshold <PATH-TO-TRAIN-DIR> train_80_20.txt   
python preprocess_threshold <PATH-TO-VAL-DIR> val_80_20.txt   
python preprocess_threshold <PATH-TO-TEST-DIR> test_80_20.txt   
```

## Train classifiers

```bash
python binary_classification.py <VECTORIZER> output.pkl
```
<VECTORIZER> can be one of {'tfidf', 'count', 'tfidf\_length', 'count\_length', 'bert'}

For lstm:
```bash
python train_lstm.py
```


## Train rankers
```bash
python train_ranknet.py <VECTORIZER> model.pt
```
<VECTORIZER> can be one of {'tfidf', 'count', 'tfidf\_length', 'count\_length', 'bert'}

For lstm:
```bash
python train_ranknet_lstm.py
```


## Misc  
Scripts in the misc directory are self explanatory.
