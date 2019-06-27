import sys
from flor.__main__ import main

examples_dir = 'tests/examples/'

def test_main():
    sys.argv = ['', 'python', examples_dir + 'iris_raw.py', 'main_test_exp_name']
    main()

