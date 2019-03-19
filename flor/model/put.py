from flor.constants import *
import flor.utils as utils
import json

def put(k, v):
    utils.cond_mkdir(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, k), 'w') as f:
        json.dump(v, f)

