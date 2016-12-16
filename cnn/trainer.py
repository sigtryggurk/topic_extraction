from model import MultiLabelCNNModelBuilder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from ast import literal_eval
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
from tools import * 

import os
import csv
import numpy as np
import argparse
import sys, pickle, os
import config as cfg

"""
  This file trains a multi-label CNN using pre-trained GloVe vectors
  on our 5_wikipedia data set.

  GloVe vectors can be downloaded http://nlp.stanford.edu/projects/glove/

"""

def train(model=None, tokenizer=None):
  # Load the training data
  texts, targets = loadTextsAndLabels(cfg.TRAIN_DATA_PATH)
  
  # Vectorize the text samples into a 2D integer tensor
  if tokenizer == None:
    tokenizer = Tokenizer(nb_words=cfg.MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
  inputs = tokenizer.texts_to_sequences(texts)

  # Pad/Cut sequences at the maximum length 
  inputs = pad_sequences(inputs, maxlen=cfg.MAX_SEQUENCE_LENGTH)

  print('Data Tensor Shape:', inputs.shape)
  print('Labels Tensor Shape:', targets.shape)

  if model == None:
    # Get the pre-trained Glove vectors
    embeddings = getGloveEmbeddings(tokenizer.word_index)

    # Build the CNN model
    model = MultiLabelCNNModelBuilder(embeddings).build()

  # Do cross validation 
  shuffle = StratifiedShuffleSplit(n_splits=cfg.NB_SPLITS, test_size=cfg.VALIDATION_SPLIT, random_state=1337)
  for train, val in shuffle.split(inputs, targets):
    # Train model
    model.fit(inputs[train], targets[train], validation_data=(inputs[val], targets[val]), 
  	      nb_epoch=cfg.NB_EPOCHS, batch_size=cfg.BATCH_SIZE)

  return model, tokenizer
  

def test(model, tokenizer):
  print("Evaluating test data")
  # Load the test data
  texts, targets = loadTextsAndLabels(cfg.TEST_DATA_PATH)
  
  # Vectorize the text samples into a 2D integer tensor
  inputs = tokenizer.texts_to_sequences(texts)
  
  # Pad/Cut sequences at the maximum length 
  inputs = pad_sequences(inputs, maxlen=cfg.MAX_SEQUENCE_LENGTH)

  return model.evaluate(inputs, targets, batch_size=cfg.BATCH_SIZE)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-r","--restore", default=None, type=str, help="Specify the run you'd like to restore from. Set to 'latest' if you want restore the latest run.")
  parser.add_argument("-t","--train", action="store_true", default=True, help="if true, trains the model on the training set")
  parser.add_argument("-e","--test", action="store_true", default=True, help="if true, evaluates the current model on the test set")

  args = parser.parse_args()

  model = tokenizer = None
  if args.restore != None:
    if args.restore == "latest":
      print("Restoring latest run")
      model, tokenizer = getLatestRun()
    else:
      model, tokenizer = getRun(args.restore)

  if args.train:
    # Make a new run directory
    saveDir = makeSaveDir()
   
    # Train the model
    model, tokenizer = train(model, tokenizer) 
    
    # Save trained model and tokenizer
    saveModel(saveDir, model)
    saveTokenizer(saveDir, tokenizer)

  if args.test and model != None and tokenizer != None:
    scores = test(model, tokenizer)
    print(" - ".join(["%s: %.4f" % (model.metrics_names[i], score) for i,score in enumerate(scores)]))
