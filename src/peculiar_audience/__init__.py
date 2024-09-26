from ._iterable import *  # noqa: F401, F403
from ._mapping import *  # noqa: F401, F403
from ._sequence import *  # noqa: F401, F403

try:
    from .version import __version__
except ImportError:
    # version.py is generated at build time
    # if for some reason it's missing use the same fallback PDM uses
    __version__ = "0.0.0"
