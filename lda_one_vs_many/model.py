import sys
sys.path.insert(0, '../')

from ast import literal_eval
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import LatentDirichletAllocation

import numpy as np
import os, csv
import global_config as cfg

EMBEDDING_DIM = 50

class LDATopicVectorizer(object):
    def __init__(self, n_topics, n_features):          # in this case X = summaries
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                             max_features=n_features,
                                             stop_words='english')
        self.lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                             learning_method='online',
                                             learning_offset=50.,
                                             random_state=0)

    def fit(self, X, y):
        tf = self.tf_vectorizer.fit_transform(X)
        self.lda.fit(tf)
        self.printTopics(self.tf_vectorizer.get_feature_names())
        return self
    
    def transform(self, X):
        tf = self.tf_vectorizer.fit_transform(X)
        transformTable = np.array(self.lda.transform(tf))
        
        return transformTable

    def printTopics(self, feature_names, n_top_words = 20):
        for topic_idx, topic in enumerate(self.lda.components_):
            print("Topic #%d:", topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print("\n")


class Classifier(object):
  def __init__(self, topics):
    self.mlb = MultiLabelBinarizer(classes=topics).fit([topics])

  def train(self):
    summaries = []
    targets = [] # enum value of category for each article

    with open(cfg.TRAIN_DATA_PATH, "r") as dataFile:
      reader = csv.reader(dataFile)
      for summary,cats in reader:
        summaries.append(summary)
        targets.append(tuple(literal_eval(cats)))
      
    print('Number of summaries: (%d)' % (len(summaries)))

    # Transform labels
    targets = self.mlb.transform(targets)    

    textClf = Pipeline([('lda', LDATopicVectorizer(n_topics=5, n_features=50)), ('clf', OneVsRestClassifier(LinearSVC()))])
    self.classifier = textClf.fit(summaries, targets)

  def predict(self, x):
    return self.classifier.predict(x)
