import sys
sys.path.insert(0, '../')

from ast import literal_eval
from glove_lda_one_vs_many.model import Classifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import numpy as np
import os, csv
import global_config as cfg

verbose = False

classifier = Classifier(cfg.TOPICS)
print("Training")
classifier.train()

summaries = []
targets = []

print("Evaluating on Test Data")
with open(cfg.TEST_DATA_PATH,'r') as dataFile:
  reader = csv.reader(dataFile)

  for summary,cats in reader:
    summaries.append(summary)
    targets.append(tuple(literal_eval(cats)))

targets = classifier.mlb.transform(targets)    
predicted = classifier.predict(summaries)

if verbose:
  for i,p in enumerate(predicted.tolist()):
    if len(predicted) - i < 20:
    	print("%s : Predicted %s, Actual %s)" % (i, p, targets[i]))

print("GloVe+LDA OneVsMany Accuracy: %f" % accuracy_score(predicted, targets))
print("GloVe+LDA OneVsMany Precision Score (global): %f" % precision_score(predicted, targets, average='micro'))
print("GloVe+LDA OneVsMany Precision Score (weighted per label): %f" % precision_score(predicted, targets, average='weighted'))
print("GloVe+LDA OneVsMany Recall Score (global): %f" % recall_score(predicted, targets, average='micro'))
print("GloVe+LDA OneVsMany Recall Score (weighted per label): %f" % recall_score(predicted, targets, average='weighted'))
print("GloVe+LDA OneVsMany F1 Score (global): %f" % f1_score(predicted, targets, average='micro'))
print("GloVe+LDA OneVsMany F1 Score (weighted per label): %f" % f1_score(predicted, targets, average='weighted'))
