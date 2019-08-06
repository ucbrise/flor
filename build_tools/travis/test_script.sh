#!/bin/bash

python --version
pyflor_install <<EOF
Y

3.6


EOF

pytest --cov=./
