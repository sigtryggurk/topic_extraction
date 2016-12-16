import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WIKIPEDIA_DIR = os.path.join(DATA_DIR, "5_wikipedia")
TRAIN_DATA_PATH = os.path.join(WIKIPEDIA_DIR, "train-data.csv")
TEST_DATA_PATH = os.path.join(WIKIPEDIA_DIR, "test-data.csv")
GLOVE_DIR = os.path.join(DATA_DIR, "glove")
# Format for a pre-trained glove file
GLOVE_DATA_PATH = lambda dim: os.path.join(GLOVE_DIR, "glove.6B.{}d.txt".format(dim)) 

TOPICS = ["computer_science", "mathematics", "music", "film", "politics"]

