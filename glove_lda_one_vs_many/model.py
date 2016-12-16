import sys
sys.path.insert(0, '../')

from ast import literal_eval
from collections import defaultdict
from glove_one_vs_many.model import TfidfEmbeddingVectorizer
from lda_one_vs_many.model import LDATopicVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import os, csv
import global_config as cfg

EMBEDDING_DIM = 50

class Classifier(object):
  def __init__(self, topics):
    self.mlb = MultiLabelBinarizer(classes=topics).fit([topics])

  def train(self):
    summaries = []
    targets = [] # enum value of category for each article

    with open(cfg.TRAIN_DATA_PATH, "r") as dataFile:
      reader = csv.reader(dataFile)

      limit = 10000
      for summary,cats in reader:
        if limit == 0:
          break
        summaries.append(summary)
        targets.append(tuple(literal_eval(cats)))
        limit -= 1
      
    print('Number of summaries: (%d)' % (len(summaries)))

    # Transform labels
    targets = self.mlb.transform(targets)    

    # Load GloVe vectors
    with open(cfg.GLOVE_DATA_PATH(EMBEDDING_DIM), "r") as lines:
      w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:]))) for line in lines}
    
    combined_features = FeatureUnion([('glove', TfidfEmbeddingVectorizer(w2v)), \
                                      ('lda', LDATopicVectorizer(n_topics=5, n_features=50))])
    textClf = Pipeline([('combined features', combined_features), ('clf', OneVsRestClassifier(LinearSVC()))])

    self.classifier = textClf.fit(summaries, targets)

  def predict(self, x):
    return self.classifier.predict(x)
