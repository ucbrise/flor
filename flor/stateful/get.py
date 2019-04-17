from flor.constants import *
import flor.utils as utils
import json

def get(k):
    try:
        with open(os.path.join(MODEL_DIR, k), 'r') as f:
            v = json.load(f)
        return v
    except:
        return Null