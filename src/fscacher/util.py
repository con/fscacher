from collections import namedtuple
from hashlib import md5
import logging
import os
import sys
import time

lgr = logging.getLogger(__name__)


class FileFingerprint(namedtuple("FileFingerprint", "mtime_ns ctime_ns size inode")):
    @classmethod
    def for_file(cls, path):
        """Simplistic generic file fingerprinting based on ctime, mtime, and size
        """
        try:
            # we can't take everything, since atime can change, etc.
            # So let's take some
            s = os.stat(path, follow_symlinks=True)
            fprint = cls.from_stat(s)
            lgr.log(5, "Fingerprint for %s: %s", path, fprint)
            return fprint
        except Exception as exc:
            lgr.debug(f"Cannot fingerprint {path}: {exc}")

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
        self.hash = None

    def add_file(self, path, fprint: FileFingerprint):
        fprint_hash = md5(
            ascii((str(path), fprint.to_tuple())).encode("us-ascii")
        ).digest()
        if self.hash is None:
            self.hash = fprint_hash
            self.last_modified = fprint.mtime_ns
        else:
            self.hash = xor_bytes(self.hash, fprint_hash)
            if self.last_modified < fprint.mtime_ns:
                self.last_modified = fprint.mtime_ns

    def modified_in_window(self, min_dtime):
        if self.last_modified is None:
            return False
        else:
            return abs(time.time() - self.last_modified * 1e-9) < min_dtime

    def to_tuple(self):
        if self.hash is None:
            return (None,)
        else:
            return (self.hash.hex(),)


def xor_bytes(b1: bytes, b2: bytes) -> bytes:
    length = max(len(b1), len(b2))
    i1 = int.from_bytes(b1, sys.byteorder)
    i2 = int.from_bytes(b2, sys.byteorder)
    return (i1 ^ i2).to_bytes(length, sys.byteorder)
