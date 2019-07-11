import pstats
import io
import pandas as pd
import numpy as np

import os
import re
import importlib

# The module below implements a function that identifies which modules are C Extensions
import c_extens  # thanks https://stackoverflow.com/questions/20339053/in-python-how-can-one-tell-if-a-module-comes-from-a-c-extension


def get_raw_output(source):
    stream = io.StringIO()
    stats = pstats.Stats(source, stream=stream)
    stats.print_stats()
    raw_output = stream.getvalue().split('\n')
    stream.close()
    return raw_output


def to_df(raw: str) -> pd.DataFrame:
    """
    :param raw: string output of cProfile
    :return: a Pandas DataFrame
    """
    if os.path.exists(raw):
        raw = get_raw_output(raw)

    columns = ['ncalls', 'tottime', 'percall', 'cumtime', 'percall', 'filename:lineno(function)']

    # filter non-tabular metadata
    tabular_raw = None
    for i, line in enumerate(raw):
        if line.split() == columns:
            tabular_raw = raw[i:]
            break

    # we unique the names of the columns for Pandas
    columns[2] = 'percallnode'
    columns[4] = 'percallgraph'

    # we rename for pandas
    columns[5] = 'plf'  # p: path, l: lineno, f: function

    tabular = [re.split('\s\s+', line) for line in tabular_raw[1:]]

    # preprocess
    for i in range(len(tabular)):
        # filter empty cells
        line = tabular.pop(0)
        line = [word for word in line if word]
        if line:
            # Last cell needs to be unmerged
            last = line.pop()
            split_idx = last.find(' ')
            n, s = last[0:split_idx], last[split_idx + 1:]
            line.append(n)
            line.append(s)
            # assert len(line) == len(columns), "i: {}, len: {} --- {}".format(i, len(line), line)
            while len(line) != len(columns):
                line = sum([re.split(' ', x) if x and '{' not in x else re.split(' ', x, 1) if x[0] != '{' else [x] for x in line], [])
            else:
                tabular.append(line)

    # Ready to load into DF
    df = pd.DataFrame(tabular)
    df.columns = columns

    for col in ['tottime', 'cumtime']:
        df[col] = df[col].astype('float')

    return df


class Template:

    def __init__(self):
        self.special = {
            "\{method '.*' of '.*' objects\}": lambda x: (x.split("'")[3], x.split("'")[1]),
            "\{built-in method .*\}": lambda x: (None, x[1:-1].split()[-1]),
            "\{function (.*)\.(.*) at 0x(\w)*\}": lambda x: (x[1:-1].split()[1].split('.')),
            "<frozen (.*)\.(.*)>:(\d+)\(\w*\)": lambda x: (
                x.split(':')[0][len('<frozen') + 1:-1], x.split(':')[1].split('(')[1][:-1]),
            "<frozen (.*)\.(.*)>:(\d+)\(<\w*>\)": lambda x: (x.split(':')[0][len('<frozen') + 1:-1], None),
            "\w+\.py:\d+\(\w+\)": lambda x: (None, None),
            "\w+\.py:\d+\(<\w+>\)": lambda x: (None, None),
            "<string>:\d+\(<\w+>\)": lambda x: (None, None),
            "<string>:\d+\(\w+\)": lambda x: (None, None),
            "\{instance\}": lambda x: (None, None),
            "\{((\w)+\.)+\w+\}": lambda x: (x[0:-1], None),
            "<decorator-gen-\d+>:\d+\(<?\w+>?\)": lambda x: (None, None)
        }

    def validate(self, df):
        flag = False

        def setf():
            nonlocal flag
            flag = True
            return True

        for line in df['plf']:
            k = self.match(line)
            not k and '/' != line[0] and setf() and print(line)
        if not flag:
            print("Validation success.")
        else:
            print("Validation failed.")

    def match(self, t):
        for k in self.special:
            pattern = re.compile(k)
            if pattern.match(t):
                return k

    def is_c_extension(self, t):
        k = self.match(t)
        if k is not None:
            m, f = self.special[k](t)  # m: module, f: function
            if m is None:
                return False
            ms = m.split('.')
            for i in range(len(ms)):
                candidate_mod_name = '.'.join(ms[:len(ms) - i])
                try:
                    candidate_mod = importlib.import_module(candidate_mod_name)
                except:
                    continue
                try:
                    ans = c_extens.is_c_extension(candidate_mod)  # I am trusting a stackoverflow solution
                    if ans is None:
                        ans = False
                    return ans
                except:
                    return False
        return False


toby = Template()


def get_c_df(df):
    c_df = df[df['plf'].map(toby.is_c_extension)]
    c_df.ncalls = c_df.ncalls.astype('int')
    return c_df


def get_c_fraction(df):
    if os.path.exists(df):
        df = to_df(df)
    c_df = get_c_df(df)
    # return "{0:.3%}".format(c_df['cumtime'].map(float).sum() / df['cumtime'].map(float).max())
    return c_df['cumtime'].map(float).sum() / df['cumtime'].map(float).max()
