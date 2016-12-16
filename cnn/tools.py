from ast import literal_eval
from datetime import datetime
from keras.models import load_model

import pickle, os, csv
import numpy as np
import config as cfg

"""
  Various Tools used by the trainer to read, transform and save data.

"""

def loadTextsAndLabels(dataPath, limit=None):
  print("Loading data from %s" % dataPath)
  texts = []
  labels = []
  with open(cfg.TRAIN_DATA_PATH, encoding='utf-8') as dataFile:
    reader = csv.reader(dataFile)
    for summary, categories in reader:
      texts.append(summary)
      labels.append(tuple(literal_eval(categories)))
      
      if limit != None:
        limit -= 1
        if limit == 0: break

  labels = np.array(cfg.LABELER.transform(labels))
  return texts, labels 

def getGloveEmbeddings(word_index):
  print("Initializing GloVe Embeddings")

  w2v = {}
  with open(cfg.GLOVE_DATA_PATH(cfg.EMBEDDING_DIM)) as gloveFile:
    for line in gloveFile:
      values = line.split()
      word = values[0]
      vec = np.asarray(values[1:], dtype='float32')
      w2v[word] = vec

    nb_words = min(cfg.MAX_NB_WORDS, len(word_index)) 
    glove_matrix = np.zeros((nb_words + 1, cfg.EMBEDDING_DIM)) 
    for word, i in word_index.items(): 
      if i > cfg.MAX_NB_WORDS: 
        continue 
      vec = w2v.get(word) 
      if vec is not None: 
        glove_matrix[i] = vec
  
  return glove_matrix

def getRun(loadDir):
    try:
      print("Restoring model from run %s" % loadDir)
      return loadModel(loadDir), loadTokenizer(loadDir)
    except Exception:
      raise ValueError("Not a valid run") 

def getLatestRun():
  candidates = {}
  for d in os.listdir(cfg.RUNS_DIR):
    try:
      candidates[datetime.fromtimestamp(float(d))] = d
    except Exception:
      continue # Not a valid run directory 
  
  loadDir = candidates[max(candidates.keys())]
  print("Restoring model from run %s" % loadDir)
  return loadModel(loadDir), loadTokenizer(loadDir)

def makeSaveDir():
  saveDir = str(datetime.now().timestamp())
  fullSaveDir = os.path.join(cfg.RUNS_DIR, saveDir)
  if not os.path.exists(fullSaveDir):
    os.makedirs(fullSaveDir)
  return saveDir

def saveModel(saveDir, model):
  print("Saving model from run %s" % saveDir)
  modelPath = os.path.join(cfg.RUNS_DIR, saveDir, cfg.MODEL_NAME)
  model.save(modelPath)

def loadModel(loadDir):
  modelPath = os.path.join(cfg.RUNS_DIR, loadDir, cfg.MODEL_NAME)
  return load_model(modelPath)

def saveTokenizer(saveDir, tokenizer):
  print("Saving tokenizer from run %s" % saveDir)
  tokenizerPath = os.path.join(cfg.RUNS_DIR, saveDir, cfg.TOKENIZER_NAME)
  with open(tokenizerPath, "wb") as tokenizerFile:
    pickle.dump(tokenizer, tokenizerFile)

def loadTokenizer(loadDir):
  tokenizerPath = os.path.join(cfg.RUNS_DIR, loadDir, cfg.TOKENIZER_NAME)
  with open(tokenizerPath, "rb") as tokenizerFile:
    return pickle.load(tokenizerFile)
