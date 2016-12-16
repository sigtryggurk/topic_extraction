# Setup
  First make sure your modules are the right version (check out requirements.txt).
  
  NOTE: We are using python 3 throughout

## Downloading Data

  * Download the 5\_wikipedia data set from:
      https://drive.google.com/drive/folders/0Bw3qoCCWu7Mfbldlb1dvUHZ4UHM

  * Download the pre-trained GloVe vectors from:
      http://nlp.stanford.edu/projects/glove/

## Data directories

  Data paths are customizable using `global_config.py`, but default locations are:

  *  `5_wikipedia` data set goes into `data/5_wikipedia`

  * Pre-trained GloVe vectors go into `data/glove`
  
# Training a Model

  For every `model.py` there's a corresponding `trainer.py`
  Simply run:
	`$ python3 trainer.py` (CNN also has some command line flags)
  
  and you will train the model and evaluate it using the test data 
