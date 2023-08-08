import json

def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False
        
def duck_cast(v, default):
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    elif isinstance(default, float):
        return float(v)
    elif isinstance(default, str):
        return str(v) if v else ""
    elif isinstance(default, list):
        return list(v) if v else []
    elif isinstance(default, tuple):
        return tuple(v) if v else tuple([])
    elif isinstance(default, bytes):
        return bytes(v)
    elif isinstance(default, bytearray):
        return bytearray(v)
    else:
        raise TypeError(f"Unsupported type: {type(default)}")