
"""
  This file defines the default hyper-parameters for the Multi-label CNN model
"""

OPTIMIZER = 'rmsprop' # TODO(siggi) list possible optimizers
ACTIVATION = 'relu'
NB_FILTER = 128
NB_CONVPOOL_LAYERS = 1
FILTER_LENS = [5] # Length of each filter, size must match NB_CONVPOOL_LAYERS
POOL_LENS = [35] # Length of each max pool, size must match NB_CONVPOOL_LAYERS
DROPOUT_PROB = 0
