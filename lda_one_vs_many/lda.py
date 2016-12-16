from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

import os
from wikipedia.tools import WikipediaClient, WikipediaPage, UnicodeReader
from ast import literal_eval
import numpy as np

n_features = 1000
n_topics = 5
n_top_words = 20

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print() 


# Loading data
print("Loading dataset...")
t0 = time()

summaries = []
targets = [] # enum value of category for each article

dataPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wikipedia/data", "wikipedia-data.csv")
with open(dataPath, "rb") as dataFile:
    reader = UnicodeReader(dataFile)

    for summary,cats in reader:
        summaries.append(summary)
        targets.append(tuple(literal_eval(cats)))

n_samples = len(summaries)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(summaries)
print("done in %0.3fs." % (time() - t0))

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
doc_topic_dist = np.array(lda.transform(tf))
doc_topic_dist2 = np.array(lda.transform(tf))
doc_topic = np.hstack((doc_topic_dist, doc_topic_dist2))
print(doc_topic_dist.shape)
print(doc_topic_dist2.shape)
print(doc_topic.shape)
