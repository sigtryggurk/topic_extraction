import sys
sys.path.insert(0, '../')

from ast import literal_eval
from baseline.model import Classifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import os, csv
import global_config as cfg

# Set this to True if you want to see every prediction
verbose = False

classifier = Classifier()
print("Training")
classifier.train()

summaries = []
targets = []

print("Evaluating on test data")
with open(cfg.TEST_DATA_PATH,'r') as dataFile:
  reader = csv.reader(dataFile)

  for data in reader:
    summaries.append(data[0])
    category = ','.join(literal_eval(data[1]))

    if verbose:
      print(category)

    targets.append(classifier.categoryDict.get(category))

predicted = classifier.predict(summaries)

if verbose:
  for i,p in enumerate(predicted.tolist()):
    print("Predicted %s, Actual %s)" % (p, targets[i]))
print("Baseline Accuracy: %f" % accuracy_score(predicted, targets))
print("Baseline Precision Score (global): %f" % precision_score(predicted, targets, average='micro'))
print("Baseline Precision Score (weighted per label): %f" % precision_score(predicted, targets, average='weighted'))
print("Baseline Recall Score (global): %f" % recall_score(predicted, targets, average='micro'))
print("Baseline Recall Score (weighted per label): %f" % recall_score(predicted, targets, average='weighted'))
print("Baseline F1 Score (global): %f" % f1_score(predicted, targets, average='micro'))
print("Baseline F1 Score (weighted per label): %f" % f1_score(predicted, targets, average='weighted'))
