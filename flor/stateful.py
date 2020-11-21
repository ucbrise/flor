from flor.constants import *

MODE = EXEC
NAME = None
LOG_PATH = None
LOG_DATA_PATH = None
MEMO_PATH = None
PRED_INIT_MODE = None # Predecessor Initialization Mode: for Parallel/Sampling replay
PSEUDORESUMING = False

PID = None
NPARTS = None

# For Adaptive Checkpointing
iterations_count = 0
period = -1
# whether we are fine-tuning or pretraining. Boolean setting.
pretraining = False
# For re-exec. Tells us which global keys have RBRACKETS instead of values
# A global key has an RBRACKET if adaptive checkpointing policy chooses not to serialize args to proc_side_effects
rbracket_gk = set([])

# FOR AUTOPARALLELISM
outermost_sk = None

