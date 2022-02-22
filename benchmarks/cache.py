from abc import ABC, abstractmethod
from hashlib import sha256
import os
from pathlib import Path
import random
from time import sleep, time
from uuid import uuid4
from morecontext import envset
from fscacher import PersistentCache


class BaseCacheBenchmark(ABC):
    param_names = ["mode"]
    params = [["populate", "hit", "ignore"]]

    @abstractmethod
    def init_path(self, *args):
        # Must return the path created
        ...

    @abstractmethod
    def init_func(cache):
        # Must return the function
        ...

    def init_cache(self, ignore: bool = False):
        with envset("FSCACHER_CACHE", "ignore" if ignore else ""):
            self.cache = PersistentCache(path=str(uuid4()))
        self.func = self.init_func(self.cache)

    def setup(self, mode, *args):
        self.path = self.init_path(mode, *args)
        if mode == "hit":
            self.init_cache()
            self.func(self.path)
        elif mode == "ignore":
            self.init_cache(ignore=True)

    def time_cache(self, mode, *_args):
        if mode == "populate":
            self.init_cache()
        self.func(self.path)


class TimeFile(BaseCacheBenchmark):
    FILE_SIZE = 1024

    def init_path(self, *_args):
        with open("foo.dat", "wb") as fp:
            fp.write(bytes(random.choices(range(256), k=self.FILE_SIZE)))
        return "foo.dat"

    @staticmethod
    def init_func(cache):
        @cache.memoize_path
        def hashfile(path):
            # "emulate" slow invocation so significant raise in benchmark
            # consumed time would mean that we invoked it instead
            # of using cached value
            sleep(0.01)
            with open(path, "rb") as fp:
                return sha256(fp.read()).hexdigest()

        return hashfile


class BaseDirectoryBenchmark(BaseCacheBenchmark):
    param_names = BaseCacheBenchmark.param_names + ["tmpdir"]
    params = BaseCacheBenchmark.params + [
        os.environ.get("FSCACHER_BENCH_TMPDIRS", ".").split(":")
    ]

    @staticmethod
    @abstractmethod
    def get_layout():
        ...

    def init_path(self, _mode, tmpdir):
        dirpath = Path(tmpdir, str(uuid4()))
        dirpath.mkdir(parents=True)
        base_time = time()
        dirs = [dirpath]
        layout = self.get_layout()
        for i, width in enumerate(layout):
            if i < len(layout) - 1:
                dirs2 = []
                for d in dirs:
                    for x in range(width):
                        d2 = d / f"d{x}"
                        d2.mkdir()
                        dirs2.append(d2)
                dirs = dirs2
            else:
                for j, d in enumerate(dirs):
                    for x in range(width):
                        f = d / f"f{x}.dat"
                        f.write_bytes(b"\0" * random.randint(1, 1024))
                        t = base_time - x - j * width
                        os.utime(f, (t, t))
        return dirpath

    @staticmethod
    def init_func(cache):
        @cache.memoize_path
        def dirsize(path):
            total_size = 0
            with os.scandir(path) as entries:
                for e in entries:
                    if e.is_dir():
                        total_size += dirsize(e.path)
                    else:
                        total_size += e.stat().st_size
            return total_size

        return dirsize


class TimeFlatDirectory(BaseDirectoryBenchmark):
    @staticmethod
    def get_layout():
        return (100,)


class TimeDeepDirectory(BaseDirectoryBenchmark):
    @staticmethod
    def get_layout():
        return (3, 3, 3, 3)
