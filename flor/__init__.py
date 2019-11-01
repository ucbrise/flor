import sys
import os
from datetime import datetime

from flor.constants import *
import flor.stateful as flags
import flor.utils as utils

# Fetch the flags we need without disrupting user code
flor_settings = {
    'mode': ['exec', 'reexec'], # default: exec
    'name': ANY,
    'memo': ANY
}

argvs = []
flor_arg = None
for each in sys.argv:
    if '--flor' != each[0:len('--flor')]:
        argvs.append(each)
    else:
        flor_arg = each.split('=')[1]
        assert flor_arg != '', "[FLOR] Enter a setting and value: {}".format(flor_settings)
sys.argv = argvs

user_settings = {}

# Validate the user entered valid settings
flor_arg = flor_arg.split(',')
flor_arg = [each.split(':') for each in flor_arg]
for (k, v) in flor_arg:
    assert k in flor_settings, "[FLOR] Invalid setting: {}".format(k)
    assert flor_settings[k] is ANY or v in flor_settings[k], "[FLOR] Invalid value for setting `{}`. Value must be one of {}".format(k, flor_settings[k])
    assert k not in user_settings, "[FLOR] Duplicate setting entered: {}".format(k)
    user_settings[k] = v

# Check that required flags are set
assert 'name' in user_settings, "[FLOR] Missing required parameter: name."
flags.NAME = user_settings['name']
flags.LOG_PATH = os.path.join(os.path.expanduser('~'), '.flor', flags.NAME, "{}.json".format(datetime.now().strftime("%Y%m%d-%H%M%S")))

# Update default settings
if 'mode' in user_settings and user_settings['mode'] == 'reexec':
    assert 'memo' in user_settings, "[FLOR] On Re-execution, please specify a memoized file"
    flags.MEMO_PATH = os.path.join(os.path.expanduser('~'), '.flor', flags.NAME, user_settings['memo'])
    assert os.path.exists(flags.MEMO_PATH)
    flags.MODE = REEXEC

# Mkdirs
utils.cond_mkdir(os.path.join(os.path.expanduser('~'), '.flor'))
utils.cond_mkdir(os.path.join(os.path.expanduser('~'), '.flor', flags.NAME))

# Finish initializing flor
from flor.writer import pin_state, random_seed
from flor.skipblock import SkipBlock

SKIP = flags.MODE is REEXEC
__all__ = ['pin_state', 'random_seed', 'SKIP', 'SkipBlock']