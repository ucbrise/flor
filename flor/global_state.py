#!/usr/bin/env python3

import os

interactive = False
nb_name = None
log_name = None

# Temporary directory for code injection
ci_temporary_directory = None
mapper = None

# Global log object for append.
FLOR_DIR = os.path.join(os.path.expanduser('~'), '.flor')
FLOR_CUR = os.path.join(FLOR_DIR, '.current')
flog = None
