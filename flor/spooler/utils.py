import re

def convert_to_int(s):
    return int(s) if s.isdigit() else s
    
def natural_key(s):
    """ Turn a string into a list of string and number chunks.
        "20191213-170335.34.json" -> [20191213, "-", 170335, ".", 34, ".json"]
    """
    return [ convert_to_int(c) for c in re.split('([0-9]+)', s) ]

