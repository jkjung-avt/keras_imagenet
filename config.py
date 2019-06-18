"""config.py
"""

from easydict import EasyDict as edict

config = edict()

# Subdirectory name for saving trained weights and models
config.SAVE_DIR = 'saves'

# Subdirectory name for saving TensorBoard log files
config.LOG_DIR = 'logs'

# Number of parallel works for generating training/validation data
config.NUM_DATA_WORKERS = 4

# Do image data augmentation or not
config.DATA_AUGMENTATION = True

# Set 'amsgrad' for the Adam optimizer or not
config.ADAM_USE_AMSGRAD = False
