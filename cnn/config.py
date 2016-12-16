import sys
sys.path.insert(0, '../')

from global_config import *
from os.path import join
from sklearn.preprocessing import MultiLabelBinarizer

"""
  This file sets configurations such as location of data directories,
  location of trained models.
"""

# File Locations
RUNS_DIR = 'runs'
MODEL_NAME = 'model.h5'
TOKENIZER_NAME = 'tokenizer'

# Model Constants
METRICS = ['binary_accuracy', 'categorical_accuracy', 'precision', 'recall','fbeta_score']
VALIDATION_SPLIT = 0.2
NB_SPLITS = 1
MAX_SEQUENCE_LENGTH = 1000#1 << 10
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100 # can be 50, 100, 200, 300
NB_EPOCHS = 1
BATCH_SIZE = 128

# Fix topics and labeler so it's constant accross runs 
# TODO(siggi): Augment data to have binary labels, so we don't have to have this hack
NUM_TOPICS = len(TOPICS)
LABELER = MultiLabelBinarizer(classes=TOPICS).fit([TOPICS])
