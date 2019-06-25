import inspect, os
from importlib.machinery import ExtensionFileLoader, EXTENSION_SUFFIXES
from types import ModuleType

def is_c_extension(module: ModuleType) -> bool:
    '''
    `True` only if the passed module is a C extension implemented as a
    dynamically linked shared library specific to the current platform.

    Parameters
    ----------
    module : ModuleType
        Previously imported module object to be tested.

    Returns
    ----------
    bool
        `True` only if this module is a C extension.
    '''
    assert isinstance(module, ModuleType), '"{}" not a module.'.format(module)

    # If this module was loaded by a PEP 302-compliant CPython-specific loader
    # loading only C extensions, this module is a C extension.
    if isinstance(getattr(module, '__loader__', None), ExtensionFileLoader):
        return True

    # Else, fallback to filetype matching heuristics.
    #
    # Absolute path of the file defining this module.
    try:
        module_filename = inspect.getfile(module)

        # "."-prefixed filetype of this path if any or the empty string otherwise.
        module_filetype = os.path.splitext(module_filename)[1]

        # This module is only a C extension if this path's filetype is that of a
        # C extension specific to the current platform.
        return module_filetype in EXTENSION_SUFFIXESP
    except:
        pass

