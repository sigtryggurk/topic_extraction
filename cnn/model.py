'''
  This file contains th
'''
import numpy as np

from keras.layers.core import Dropout
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, load_model
import hyperparameters as hp
import config as cfg

class MultiLabelCNNModelBuilder(object):
  def __init__(self, embeddings):
    self.embeddings = embeddings
  
  def build(self):
    print('Building model.')
    input_dim, output_dim = self.embeddings.shape
    
    inputs = Input(shape=(cfg.MAX_SEQUENCE_LENGTH,), dtype='int32')
    x = Embedding(input_dim,
                  output_dim,
                  weights=[self.embeddings],
                  input_length=cfg.MAX_SEQUENCE_LENGTH,
                  trainable=False)(inputs)
    for i in range(hp.NB_CONVPOOL_LAYERS):
      x = Conv1D(hp.NB_FILTER, hp.FILTER_LENS[i], activation=hp.ACTIVATION)(x)
      #x = Dropout(hp.DROPOUT_PROB)(x) 
      x = MaxPooling1D(hp.POOL_LENS[i])(x)
    x = Flatten()(x)
   # x = Dense(hp.NB_FILTER, activation=hp.ACTIVATION)(x)
    preds = Dense(cfg.NUM_TOPICS, activation='sigmoid')(x)
  
    model = Model(inputs, preds)
    model.compile(loss='binary_crossentropy', optimizer=hp.OPTIMIZER, metrics=cfg.METRICS)
    return model  
