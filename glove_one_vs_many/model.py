import sys
sys.path.insert(0, '../')

from ast import literal_eval
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import os, csv
import global_config as cfg

EMBEDDING_DIM = 50

# using sci-kit learn tutorial
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = EMBEDDING_DIM 

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = EMBEDDING_DIM 

    def fit(self, X, y):
        tfidf = TfidfVectorizer()#analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

class Classifier(object):
  def __init__(self, topics):
    self.mlb = MultiLabelBinarizer(classes=topics).fit([topics])

  def train(self):
    summaries = []
    targets = [] # enum value of category for each article
    dataPath = cfg.TRAIN_DATA_PATH
    with open(dataPath, "r") as dataFile:
      reader = csv.reader(dataFile)
      for summary,cats in reader:
        summaries.append(summary)
        targets.append(tuple(literal_eval(cats)))
    
    print('Number of summaries: (%d)' % (len(summaries)))

    # Transform labels
    targets = self.mlb.transform(targets)    

    # Load GloVe vectors
    with open(cfg.GLOVE_DATA_PATH(EMBEDDING_DIM), "r") as lines:
      w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:]))) for line in lines}
    
    textClf = Pipeline([('glove', TfidfEmbeddingVectorizer(w2v)),('clf', OneVsRestClassifier(LinearSVC()))])
    self.classifier = textClf.fit(summaries, targets)


  def predict(self, x):
    return self.classifier.predict(x)
