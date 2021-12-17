from collections import deque, namedtuple
from functools import wraps
from inspect import Parameter, signature
import logging
import os
import os.path as op
import shutil
import time
import appdirs
import joblib

lgr = logging.getLogger(__name__)


class PersistentCache(object):
    """Persistent cache providing @memoize and @memoize_path decorators
    """

    _min_dtime = 0.01  # min difference between now and mtime to consider
    # for caching

    _cache_var_values = (None, "", "clear", "ignore")

    def __init__(self, name=None, tokens=None, envvar=None):
        """

        Parameters
        ----------
        name
        tokens: list of objects, optional
         To add to the fingerprint of @memoize_path (regular @memoize ATM does
         not use it).  Could be e.g. versions of relevant/used
         python modules (pynwb, etc)
        envvar: str, optional
         Name of the environment variable to query for cache settings; if not
         set, `FSCACHER_CACHE` is used
        """
        dirs = appdirs.AppDirs("fscacher")
        self._cache_file = op.join(dirs.user_cache_dir, (name or "cache"))
        self._memory = joblib.Memory(self._cache_file, verbose=0)
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

    def memoize(self, f):
        if self._ignore_cache:
            return f
        return self._memory.cache(f)

    def memoize_path(self, f):
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
        fingerprinted = self.memoize(fingerprinted)

        @wraps(f)
        def fingerprinter(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            argname, path_orig = next(iter(bound.arguments.items()))
            # we need to dereference symlinks and use that path in the function
            # call signature
            path = op.realpath(path_orig)
            if path != path_orig:
                lgr.log(5, "Dereferenced %r into %r", path_orig, path)
            if op.isdir(path):
                fprint = self._get_dir_fingerprint(path)
            else:
                fprint = self._get_file_fingerprint(path)
            bound.arguments[argname] = path
            if fprint is None:
                lgr.debug("Calling %s directly since no fingerprint for %r", f, path)
                # just call the function -- we have no fingerprint,
                # probably does not exist or permissions are wrong
                ret = f(*bound.args, **bound.kwargs)
            # We should still pass through if file was modified just now,
            # since that could mask out quick modifications.
            # Target use cases will not be like that.
            elif fprint.modified_in_window(self._min_dtime):
                lgr.debug("Calling %s directly since too short for %r", f, path)
                ret = f(*bound.args, **bound.kwargs)
            else:
                lgr.debug("Calling memoized version of %s for %s", f, path)
                # If there is a fingerprint -- inject it into the signature
                fprint_tuple = fprint.to_tuple() + (
                    tuple(self._tokens) if self._tokens else ()
                )
                ret = fingerprinted(
                    *bound.args, **{**bound.kwargs, fingerprint_kwarg: fprint_tuple}
                )
            lgr.log(1, "Returning value %r", ret)
            return ret

        # and we memoize actually that function
        return fingerprinter

    @staticmethod
    def _get_file_fingerprint(path):
        """Simplistic generic file fingerprinting based on ctime, mtime, and size
        """
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
    def _get_dir_fingerprint(path):
        fprint = DirFingerprint()
        dirqueue = deque([path])
        try:
            while dirqueue:
                d = dirqueue.popleft()
                with os.scandir(d) as entries:
                    for e in entries:
                        if e.is_dir(follow_symlinks=True):
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
        return abs(time.time() - self.mtime_ns * 1e-9) < min_dtime

    def to_tuple(self):
        return tuple(self)


class DirFingerprint:
    def __init__(self):
        self.last_modified = None
        self.tree_fprints = {}

    def add_file(self, path, fprint: FileFingerprint):
        self.tree_fprints[path] = fprint
        if self.last_modified is None or self.last_modified < fprint.mtime_ns:
            self.last_modified = fprint.mtime_ns

    def modified_in_window(self, min_dtime):
        if self.last_modified is None:
            return False
        else:
            return abs(time.time() - self.last_modified * 1e-9) < min_dtime

    def to_tuple(self):
        return sum(sorted(self.tree_fprints.items()), ())
