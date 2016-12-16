import sys
sys.path.insert(0, '../')

from oracle.model import Oracle
from ast import literal_eval
import os, csv
import global_config as cfg

# Set this to True if you want to see every prediction
verbose = False 

dataPath = cfg.TRAIN_DATA_PATH 

oracle = Oracle()

corrects = 0
total = 0
with open(dataPath,'r') as dataFile:
  reader = csv.reader(dataFile)
  for x,y in reader:
    y = literal_eval(y)
    prediction = oracle.predict(x)
    if verbose and prediction != y:
      print(u'Input: %s, Prediction: %s, Correct: %s' % (x, prediction, y))
    
    corrects += prediction == y
    total +=1
print('TOTAL %s' % total)
print("Oracle Accuracy: {0:,.2f}%".format(100 * float(corrects)/float(total)))
