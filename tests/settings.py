_TESTS_DIR = 'tests/'
_EXAMPLES_DIR = _TESTS_DIR + 'examples/'
_LOGS_DIR = _EXAMPLES_DIR + 'logs_examples/'

_DIGITS_RAW_FILE = _EXAMPLES_DIR + 'digits_raw.py'


def init():
    global test_dir
    global examples_dir
    global logs_dir

    test_dir = 'tests/'

    examples_dir = test_dir + 'examples/'

    logs_dir = examples_dir + 'logs_examples/'
