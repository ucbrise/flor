"""
Helper for transformer
"""
import re
import astor
import ast

def greeting(s):
    return "Hello {}.".format(s)

def dump(s):
    # internal
    return 'flog.serialize({})'.format(s)

def unparse(n):
    return astor.to_source(n)

def hard_strip(s):
    return s.strip().replace('\n', '')

def get_whitespace(s):
    return re.match(r"\s*", s).group()

def escape(s):
    # internal
    return s.strip().replace("'", '"')

def proc_rhs(n):
    if isinstance(n, str):
        return dump(hard_strip(n))
    else:
        return dump(hard_strip(unparse(n)))

def proc_lhs(n, esc=True):
    if isinstance(n, str):
        if esc:
            return hard_strip(escape(n))
        else:
            return hard_strip(n)
    else:
        if esc:
            return hard_strip(escape(unparse(n)))
        else:
            return hard_strip(unparse(n))

def neg(s):
    return "not ({})".format(s)