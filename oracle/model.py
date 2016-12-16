import sys
sys.path.insert(0, '../')

from ast import literal_eval
import os, csv
import global_config as cfg

class Oracle(object):
  def __init__(self):
    self.data = {}
    # All the data
    with open(cfg.TRAIN_DATA_PATH, "r") as dataFile:
      reader = csv.reader(dataFile)
      for summary, topics in reader:
        self.data[summary] = literal_eval(topics) 
    with open(cfg.TEST_DATA_PATH, "r") as dataFile:
      reader = csv.reader(dataFile)
      for summary, topics in reader:
        self.data[summary] = literal_eval(topics) 
  
  def predict(self, x):
    return self.data[x]
