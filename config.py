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

# Batch size for each iteration
config.BATCH_SIZE = 64

# Accumulate gradients and do an update for this many iterations
config.ITER_SIZE = 4

# Initial learning rate
config.INITIAL_LR = 4e-4

# Learning rate decay
config.LR_DECAY = 1e-6

# Total number of epochs for training
config.EPOCHS = 3
