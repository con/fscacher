"""Fingerprinting of git-annex'ed files by their keys"""

import os
import os.path as op
import re

#: Backends whose keys are hashes of the content, and thus pin it (with or
#: without the ``E`` suffix, which adds the file extension to the key).  Others
#: do not: ``WORM`` keys are made of size, mtime and file name, ``URL`` and
#: ``VURL`` keys of a URL whose content may change, and external ``X*``
#: backends are unknown.
CONTENT_HASH_BACKEND_RE = re.compile(
    r"(SHA(1|224|256|384|512)|SHA3_(224|256|384|512)|SKEIN(256|512)"
    r"|BLAKE2(B|BP|S|SP)\d+|MD5)E?"
)


def annex_key_fingerprint(path, *, pair_with_path=True):
    """
    Fingerprint a locked git-annex'ed file by its key, for use as the
    ``custom_fingerprint`` of `PersistentCache.memoize_path`

    Returns the key if ``path`` is a symbolic link into a git-annex object
    store (``.../annex/objects/.../KEY/KEY``) and the key's backend hashes the
    content (see `CONTENT_HASH_BACKEND_RE`); otherwise returns `None`, so that
    the file is fingerprinted by ``stat()`` as usual.  That is the case of an
    unlocked file (including files on an adjusted branch, as on Windows),
    whose key is not updated when the file is modified.  Only the link is read,
    so this is cheap, and a file whose content is not present (e.g., dropped)
    is still fingerprinted: cached results are then returned for it.

    Parameters
    ----------
    pair_with_path: bool, optional
     If true (the default), the fingerprint is the pair of the absolute path
     (as given, not dereferenced) and the key, so results are only shared by
     files at the same path, as needed when they depend on the path (e.g., on
     the extension or on neighboring files).  If false, results are shared by
     all files with the same key, e.g., across clones of a dataset.
    """
    if isinstance(path, os.DirEntry) and not path.is_symlink():
        # known from the directory listing, without a system call
        return None
    try:
        path = os.fsdecode(os.fspath(path))
        target = os.fsdecode(os.readlink(path))
    except (TypeError, ValueError, OSError):
        return None
    parts = target.replace("\\", "/").split("/")
    if len(parts) < 6 or parts[-6:-4] != ["annex", "objects"] or parts[-1] != parts[-2]:
        return None
    key = parts[-1]
    if not CONTENT_HASH_BACKEND_RE.fullmatch(key.split("-", 1)[0]):
        return None
    return (op.abspath(path), key) if pair_with_path else key
