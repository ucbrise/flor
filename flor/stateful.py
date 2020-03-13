from flor.constants import *

MODE = EXEC
NAME = None
LOG_PATH = None
LOG_DATA_PATH = None
MEMO_PATH = None

# For Adaptive Checkpointing
iterations_count = 0
period = -1
# whether we are fine-tuning or pretraining. Boolean setting.
pretraining = False