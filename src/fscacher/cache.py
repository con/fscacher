from collections import deque, namedtuple
from functools import partial, wraps
from hashlib import md5
from inspect import Parameter, signature
import logging
import os
import os.path as op
import shutil
import time
import joblib
from platformdirs import PlatformDirs

lgr = logging.getLogger(__name__)


class PersistentCache:
    """Persistent cache providing @memoize and @memoize_path decorators"""

    _min_dtime = 0.01  # min difference between now and mtime to consider
    # for caching

    _cache_var_values = (None, "", "clear", "ignore")

    def __init__(self, name=None, *, path=None, tokens=None, envvar=None):
        """
        Parameters
        ----------
        name: str, optional
         Basename for the directory in which to store the cache.  Mutually
         exclusive with `path`.
        path: str or pathlib.Path, optional
         Directory path at which to store the cache.  If not specified, the
         cache is stored in `USER_CACHE/fscacher/NAME`, where NAME is the value
         of the `name` parameter (default: "cache").  Mutually exclusive with
         `name`.
        tokens: list of objects, optional
         To add to the fingerprint of @memoize_path (regular @memoize ATM does
         not use it).  Could be e.g. versions of relevant/used
         python modules (pynwb, etc)
        envvar: str, optional
         Name of the environment variable to query for cache settings; if not
         set, `FSCACHER_CACHE` is used
        """
        if path is None:
            dirs = PlatformDirs("fscacher")
            path = op.join(dirs.user_cache_dir, (name or "cache"))
        elif name is not None:
            raise ValueError("'name' and 'path' are mutually exclusive")
        self._memory = joblib.Memory(path, verbose=0)
        cntrl_value = None
        if envvar is not None:
            cntrl_var = envvar
            cntrl_value = os.environ.get(cntrl_var)
        if cntrl_value is None:
            cntrl_var = "FSCACHER_CACHE"
            cntrl_value = os.environ.get(cntrl_var)
        if cntrl_value not in self._cache_var_values:
            lgr.warning(
                f"{cntrl_var}={cntrl_value} is not understood and thus ignored."
                f" Known values are {self._cache_var_values}"
            )
        if cntrl_value == "clear":
            self.clear()
        self._ignore_cache = cntrl_value == "ignore"
        self._tokens = tokens

    def clear(self):
        try:
            self._memory.clear(warn=False)
        except Exception as exc:
            lgr.debug("joblib failed to clear its cache: %s", exc)

        # and completely destroy the directory
        try:
            if op.exists(self._memory.location):
                shutil.rmtree(self._memory.location)
        except Exception as exc:
            lgr.warning(f"Failed to clear out the cache directory: {exc}")

    def memoize(self, f=None, *, exclude_kwargs=None):
        if f is None:
            return partial(self.memoize, exclude_kwargs=exclude_kwargs)
        if self._ignore_cache:
            return f
        return self._memory.cache(f, ignore=exclude_kwargs)

    def memoize_path(self, f=None, *, exclude_kwargs=None, custom_fingerprint=None):
        """
        Memoize a function whose first argument is a path, keyed on a
        fingerprint of the file or directory at that path

        Parameters
        ----------
        exclude_kwargs: list of str, optional
         Names of arguments of the decorated function to ignore for caching
         purposes
        custom_fingerprint: callable, optional
         An alternative to the built-in ``stat()``-based fingerprint, e.g.
         `fscacher.annex.annex_key_fingerprint`.  It is called with the value
         of the first argument, and with the path of each entry met while
         fingerprinting a directory, and returns either a fingerprint of it or
         `None` to fall back to ``stat()``.

         A fingerprint must change whenever the result may change (there is
         no "modified just now" window for it), and must be picklable and have
         a stable ``repr()`` (e.g., a string or a tuple of strings).  Results
         are shared between all values with equal fingerprints, wherever they
         are: include the path in the fingerprint unless the result does not
         depend on it.  As the value itself is not part of the cache key, it
         need not be a path at all.
        """
        if f is None:
            return partial(
                self.memoize_path,
                exclude_kwargs=exclude_kwargs,
                custom_fingerprint=custom_fingerprint,
            )
        if self._ignore_cache:
            return f

        # we need to actually decorate a function
        fingerprint_kwarg = "_cache_fingerprint"

        @wraps(f)  # important, so .memoize correctly caches different `f`
        def fingerprinted(path, *args, **kwargs):
            _ = kwargs.pop(fingerprint_kwarg)  # discard
            lgr.debug("Running original %s on %r", f, path)
            return f(path, *args, **kwargs)

        # We need to add the fingerprint_kwarg to fingerprinted's signature so
        # that joblib doesn't complain:
        sig = signature(fingerprinted)
        fp_kwarg_param = Parameter(
            fingerprint_kwarg, Parameter.KEYWORD_ONLY, default=None
        )
        sig2 = sig.replace(
            parameters=tuple(sig.parameters.values()) + (fp_kwarg_param,)
        )
        fingerprinted.__signature__ = sig2

        path_arg = next(iter(sig.parameters.keys()))

        # we need to ignore 'path' since we would like to dereference if symlink
        # but then expect joblib's caching work on both original and dereferenced
        # So we will add dereferenced path into fingerprint_kwarg
        fingerprinted = self.memoize(
            fingerprinted,
            exclude_kwargs=[path_arg]
            + (list(exclude_kwargs) if exclude_kwargs is not None else []),
        )

        @wraps(f)
        def fingerprinter(*args, **kwargs):
            # we need to dereference symlinks and use that path in the function
            # call signature
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            path_orig = bound.arguments[path_arg]
            fprint = self._get_fingerprint(path_orig, custom_fingerprint)
            if fprint is None:
                lgr.debug(
                    "Calling %s directly since no fingerprint for %r", f, path_orig
                )
                # just call the function -- we have no fingerprint, probably
                # not a path, does not exist, or permissions are wrong
                ret = f(*args, **kwargs)
            # We should still pass through if file was modified just now,
            # since that could mask out quick modifications.
            # Target use cases will not be like that.
            elif fprint.modified_in_window(self._min_dtime):
                lgr.debug("Calling %s directly since too short for %r", f, path_orig)
                ret = f(*args, **kwargs)
            else:
                lgr.debug("Calling memoized version of %s for %r", f, path_orig)
                # If there is a fingerprint -- inject it into the signature
                kwargs_ = kwargs.copy()
                kwargs_[fingerprint_kwarg] = fprint.to_tuple() + (
                    tuple(self._tokens) if self._tokens else ()
                )
                ret = fingerprinted(*args, **kwargs_)
            lgr.log(1, "Returning value %r", ret)
            return ret

        # and we memoize actually that function
        return fingerprinter

    def _get_fingerprint(self, value, custom_fingerprint=None):
        """
        Fingerprint of ``value``: the custom one if any, else one of the file
        or directory at its dereferenced path, or `None`
        """
        if custom_fingerprint is not None:
            custom = custom_fingerprint(value)
            if custom is not None:
                lgr.log(5, "Custom fingerprint for %r: %r", value, custom)
                return CustomFingerprint(custom)
        try:
            path = op.realpath(value)
        except TypeError:
            lgr.debug("Cannot fingerprint %r: not a path-like object", value)
            return None
        if path != value:
            lgr.log(5, "Dereferenced %r into %r", value, path)
        if op.isdir(path):
            fprint = self._get_dir_fingerprint(path, custom_fingerprint)
        else:
            fprint = self._get_file_fingerprint(path)
        return None if fprint is None else PathFingerprint(path, fprint)

    @staticmethod
    def _get_file_fingerprint(path):
        """Simplistic generic file fingerprinting based on ctime, mtime, and size"""
        try:
            # we can't take everything, since atime can change, etc.
            # So let's take some
            s = os.stat(path, follow_symlinks=True)
            fprint = FileFingerprint.from_stat(s)
            lgr.log(5, "Fingerprint for %s: %s", path, fprint)
            return fprint
        except Exception as exc:
            lgr.debug(f"Cannot fingerprint {path}: {exc}")

    @staticmethod
    def _get_dir_fingerprint(path, custom_fingerprint=None):
        fprint = DirFingerprint()
        dirqueue = deque([path])
        try:
            while dirqueue:
                d = dirqueue.popleft()
                with os.scandir(d) as entries:
                    for e in entries:
                        custom = (
                            custom_fingerprint(e.path)
                            if custom_fingerprint is not None
                            else None
                        )
                        if custom is not None:
                            fprint.add_custom(e.path, custom)
                        elif e.is_dir(follow_symlinks=True):
                            dirqueue.append(e.path)
                        else:
                            s = e.stat(follow_symlinks=True)
                            fprint.add_file(e.path, FileFingerprint.from_stat(s))
        except Exception as exc:
            lgr.debug(f"Cannot fingerprint {path}: {exc}")
            return None
        else:
            return fprint


class FileFingerprint(namedtuple("FileFingerprint", "mtime_ns ctime_ns size inode")):
    @classmethod
    def from_stat(cls, s):
        return cls(s.st_mtime_ns, s.st_ctime_ns, s.st_size, s.st_ino)

    def modified_in_window(self, min_dtime):
        return abs(elapsed_since(self.mtime_ns * 1e-9)) < min_dtime

    def to_tuple(self):
        return tuple(self)


class PathFingerprint:
    """A fingerprint of a file or directory, together with its path"""

    def __init__(self, path, fprint):
        self.path = path
        self.fprint = fprint

    def modified_in_window(self, min_dtime):
        return self.fprint.modified_in_window(min_dtime)

    def to_tuple(self):
        return (self.path,) + self.fprint.to_tuple()


class CustomFingerprint:
    """A fingerprint returned by a ``custom_fingerprint`` callable"""

    def __init__(self, value):
        self.value = value

    def modified_in_window(self, min_dtime):  # noqa: U100
        # it is up to the callable to change the fingerprint on modification
        return False

    def to_tuple(self):
        # a path key starts with an absolute path, never with this marker
        return ("custom", self.value)


class DirFingerprint:
    def __init__(self):
        self.last_modified = None
        self.hash = None

    def add_file(self, path, fprint: FileFingerprint):
        self._add_hash(
            md5(ascii((str(path), fprint.to_tuple())).encode("us-ascii")).digest()
        )
        if self.last_modified is None or self.last_modified < fprint.mtime_ns:
            self.last_modified = fprint.mtime_ns

    def add_custom(self, path, value):
        self._add_hash(
            md5(ascii((str(path), ("custom", value))).encode("us-ascii")).digest()
        )

    def _add_hash(self, fprint_hash):
        self.hash = (
            fprint_hash if self.hash is None else xor_bytes(self.hash, fprint_hash)
        )

    def modified_in_window(self, min_dtime):
        if self.last_modified is None:
            return False
        else:
            return abs(elapsed_since(self.last_modified * 1e-9)) < min_dtime

    def to_tuple(self):
        if self.hash is None:
            return (None,)
        else:
            return (self.hash.hex(),)


def xor_bytes(b1: bytes, b2: bytes) -> bytes:
    length = max(len(b1), len(b2))
    # force 'little' byte order to match our assumptions on how to
    # treat bytes of different length.
    i1 = int.from_bytes(b1, "little")
    i2 = int.from_bytes(b2, "little")
    return (i1 ^ i2).to_bytes(length, "little")


def elapsed_since(t: float) -> float:
    t_now = time.time()
    dt = t_now - t
    if dt < 0:
        lgr.debug("Time is in the future: %f; now: %f; dt=%g", t, t_now, dt)
    return dt
