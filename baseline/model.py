import sys
sys.path.insert(0, '../')

from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import os, csv
import global_config as cfg

# using sci-kit learn tutorial
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# Set this to True if you want to see every prediction
verbose = False

class Classifier(object):
  def train(self):
    summaries = []
    targets = [] # enum value of category for each article
    self.categoryDict = dict() # enum of categories

    dataPath  = cfg.TRAIN_DATA_PATH
    with open(dataPath, "r") as dataFile:
      i = 0

      reader = csv.reader(dataFile)

      for summary,cats in reader:
        summaries.append(summary)
        category = ','.join(literal_eval(cats))

        if verbose:
          print(category)

        if category not in self.categoryDict:
          self.categoryDict[category]=i
          i += 1

        targets.append(self.categoryDict.get(category))

    print('Number of summaries: (%d); Number of unique categories (%d)' % (len(summaries), len(self.categoryDict.keys())))
    textClf = Pipeline([('tfidf', TfidfVectorizer()),('clf', MultinomialNB())])
    self.classifier = textClf.fit(summaries, targets)


  def predict(self, x):
    return self.classifier.predict(x)
