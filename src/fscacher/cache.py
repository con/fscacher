from functools import wraps
from inspect import Parameter, signature
import logging
import os
import os.path as op
import shutil
import appdirs
import joblib
from .fastio import walk
from .util import DirFingerprint, FileFingerprint

lgr = logging.getLogger(__name__)

DEFAULT_THREADS = 60


class PersistentCache(object):
    """Persistent cache providing @memoize and @memoize_path decorators
    """

    _min_dtime = 0.01  # min difference between now and mtime to consider
    # for caching

    _cache_var_values = (None, "", "clear", "ignore")

    def __init__(
        self, name=None, *, path=None, tokens=None, envvar=None, walk_threads=None
    ):
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
        walk_threads: int, optional
         Number of threads to use when traversing directory hierarchies
        """
        if path is None:
            dirs = appdirs.AppDirs("fscacher")
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
        self._walk_threads = walk_threads or DEFAULT_THREADS

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

    def memoize(self, f, ignore=None):
        if self._ignore_cache:
            return f
        return self._memory.cache(f, ignore=ignore)

    def memoize_path(self, f):
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
        fingerprinted = self.memoize(fingerprinted, ignore=[path_arg])

        @wraps(f)
        def fingerprinter(*args, **kwargs):
            # we need to dereference symlinks and use that path in the function
            # call signature
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            path_orig = bound.arguments[path_arg]
            path = op.realpath(path_orig)
            if path != path_orig:
                lgr.log(5, "Dereferenced %r into %r", path_orig, path)
            if op.isdir(path):
                fprint = self._get_dir_fingerprint(path)
            else:
                fprint = FileFingerprint.for_file(path)
            if fprint is None:
                lgr.debug("Calling %s directly since no fingerprint for %r", f, path)
                # just call the function -- we have no fingerprint,
                # probably does not exist or permissions are wrong
                ret = f(*args, **kwargs)
            # We should still pass through if file was modified just now,
            # since that could mask out quick modifications.
            # Target use cases will not be like that.
            elif fprint.modified_in_window(self._min_dtime):
                lgr.debug("Calling %s directly since too short for %r", f, path)
                ret = f(*args, **kwargs)
            else:
                lgr.debug("Calling memoized version of %s for %s", f, path)
                # If there is a fingerprint -- inject it into the signature
                kwargs_ = kwargs.copy()
                kwargs_[fingerprint_kwarg] = (
                    (path,)
                    + fprint.to_tuple()
                    + (tuple(self._tokens) if self._tokens else ())
                )
                ret = fingerprinted(*args, **kwargs_)
            lgr.log(1, "Returning value %r", ret)
            return ret

        # and we memoize actually that function
        return fingerprinter

    def _get_dir_fingerprint(self, dirpath):
        dprint = DirFingerprint()
        for path, fprint in walk(dirpath, threads=self._walk_threads):
            dprint.add_file(path, fprint)
        return dprint
