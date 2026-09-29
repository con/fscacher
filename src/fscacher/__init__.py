"""
Caching results of operations on heavy file trees

Visit <https://github.com/con/fscacher> for more information.
"""

from ._version import get_versions
from .annex import annex_key_fingerprint
from .cache import PersistentCache

__version__ = get_versions()["version"]
__author__ = "Center for Open Neuroscience"
__author_email__ = "debian@onerussian.com"
__license__ = "MIT"
__url__ = "https://github.com/con/fscacher"

__all__ = ["PersistentCache", "annex_key_fingerprint"]
