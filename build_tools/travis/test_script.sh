#!/bin/bash

python --version
python -c "import flor"
python -c "import flor; flor.install()" <<EOF
Y

3.6


EOF

pytest --cov=./
