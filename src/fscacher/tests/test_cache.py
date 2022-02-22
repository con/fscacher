import os
import os.path as op
import platform
import shutil
import subprocess
import sys
import time
import pytest
from .. import PersistentCache
from ..cache import DirFingerprint, FileFingerprint

platform_system = platform.system().lower()
on_windows = platform_system == "windows"


@pytest.fixture(autouse=True)
def capture_all_logs(caplog):
    caplog.set_level(1, logger="fscacher")


@pytest.fixture(scope="function")
def cache(tmp_path_factory):
    return PersistentCache(path=tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="function")
def cache_tokens(tmp_path_factory):
    return PersistentCache(path=tmp_path_factory.mktemp("cache"), tokens=["0.0.1", 1])


def test_memoize(cache):
    # Simplest testing to start with, not relying on persisting across
    # independent processes
    _comp = []

    @cache.memoize
    def f1(flag=False):
        if flag:
            raise ValueError("Got flag")
        if _comp:
            raise RuntimeError("Must not be recomputed")
        _comp.append(1)
        return 1

    assert f1() == 1
    assert f1() == 1

    # Now with some args
    _comp = []

    @cache.memoize
    def f2(*args):
        if args in _comp:
            raise RuntimeError("Must not be recomputed")
        _comp.append(args)
        return sum(args)

    assert f2(1) == 1
    assert f2(1) == 1
    assert f2(1, 2) == 3
    assert f2(1, 2) == 3
    assert _comp == [(1,), (1, 2)]


def test_memoize_multiple(cache):

    # Make sure that with the same cache can cover multiple functions
    @cache.memoize
    def f1():
        return 1

    @cache.memoize
    def f2():
        return 2

    @cache.memoize
    def f3():  # nesting call into f2
        return f2() + 1

    for _ in range(3):
        assert f1() == 1
        assert f2() == 2
        assert f3() == 3


def test_memoize_path(cache, tmp_path):
    calls = []

    @cache.memoize_path
    def memoread(path, arg, kwarg=None):
        calls.append([path, arg, kwarg])
        with open(path) as f:
            return f.read()

    def check_new_memoread(arg, content, expect_new=False):
        ncalls = len(calls)
        assert memoread(path, arg) == content
        assert len(calls) == ncalls + 1
        assert memoread(path, arg) == content
        assert len(calls) == ncalls + 1 + int(expect_new)

    fname = "file.dat"
    path = str(tmp_path / fname)

    with pytest.raises(IOError):
        memoread(path, 0)
    # and again
    with pytest.raises(IOError):
        memoread(path, 0)
    assert len(calls) == 2

    with open(path, "w") as f:
        f.write("content")

    t0 = time.time()
    try:
        # unless this computer is too slow -- there should be less than
        # cache._min_dtime between our creating the file and testing,
        # so we would force a direct read:
        check_new_memoread(0, "content", True)
    except AssertionError:  # pragma: no cover
        # if computer is indeed slow (happens on shared CIs) we might fail
        # because distance is too short
        if time.time() - t0 < cache._min_dtime:
            raise  # if we were quick but still failed -- legit
    assert calls[-1] == [path, 0, None]

    # but if we sleep - should memoize
    time.sleep(cache._min_dtime * 1.1)
    check_new_memoread(1, "content")

    # and if we modify the file -- a new read
    time.sleep(cache._min_dtime * 1.1)
    with open(path, "w") as f:
        f.write("Content")
    ncalls = len(calls)
    assert memoread(path, 1) == "Content"
    assert len(calls) == ncalls + 1

    time.sleep(cache._min_dtime * 1.1)
    check_new_memoread(0, "Content")

    # Check that symlinks should be dereferenced
    if not on_windows or sys.version_info[:2] >= (3, 8):
        # realpath doesn't work right on Windows on pre-3.8 Python, so skip the
        # test then.
        symlink1 = str(tmp_path / (fname + ".link1"))
        try:
            os.symlink(fname, symlink1)
        except OSError:
            pass
        if op.islink(symlink1):  # hopefully would just skip Windows if not supported
            ncalls = len(calls)
            assert memoread(symlink1, 0) == "Content"
            assert len(calls) == ncalls  # no new call

    # and if we "clear", would it still work?
    cache.clear()
    check_new_memoread(1, "Content")


def test_memoize_path_dir(cache, tmp_path):
    calls = []

    @cache.memoize_path
    def memoread(path, arg, kwarg=None):
        calls.append([path, arg, kwarg])
        total_size = 0
        with os.scandir(path) as entries:
            for e in entries:
                if e.is_file():
                    total_size += e.stat().st_size
        return total_size

    def check_new_memoread(arg, content, expect_new=False):
        ncalls = len(calls)
        assert memoread(path, arg) == content
        assert len(calls) == ncalls + 1
        assert memoread(path, arg) == content
        assert len(calls) == ncalls + 1 + int(expect_new)

    fname = "foo"
    path = tmp_path / fname

    with pytest.raises(IOError):
        memoread(path, 0)
    # and again
    with pytest.raises(IOError):
        memoread(path, 0)
    assert len(calls) == 2

    path.mkdir()
    (path / "a.txt").write_text("Alpha")
    (path / "b.txt").write_text("Beta")

    t0 = time.time()
    try:
        # unless this computer is too slow -- there should be less than
        # cache._min_dtime between our creating the file and testing,
        # so we would force a direct read:
        check_new_memoread(0, 9, True)
    except AssertionError:  # pragma: no cover
        # if computer is indeed slow (happens on shared CIs) we might fail
        # because distance is too short
        if time.time() - t0 < cache._min_dtime:
            raise  # if we were quick but still failed -- legit
    assert calls[-1] == [path, 0, None]

    # but if we sleep - should memoize
    time.sleep(cache._min_dtime * 1.1)
    check_new_memoread(1, 9)

    # and if we modify the file -- a new read
    time.sleep(cache._min_dtime * 1.1)
    (path / "c.txt").write_text("Gamma")
    ncalls = len(calls)
    assert memoread(path, 1) == 14
    assert len(calls) == ncalls + 1

    time.sleep(cache._min_dtime * 1.1)
    check_new_memoread(0, 14)

    # Check that symlinks should be dereferenced
    if not on_windows or sys.version_info[:2] >= (3, 8):
        # realpath doesn't work right on Windows on pre-3.8 Python, so skip the
        # test then.
        symlink1 = str(tmp_path / (fname + ".link1"))
        try:
            os.symlink(fname, symlink1)
        except OSError:
            pass
        if op.islink(symlink1):  # hopefully would just skip Windows if not supported
            ncalls = len(calls)
            assert memoread(symlink1, 0) == 14
            assert len(calls) == ncalls  # no new call

    # and if we "clear", would it still work?
    cache.clear()
    check_new_memoread(1, 14)


def test_memoize_path_persist(tmp_path):
    from subprocess import PIPE, run

    script = tmp_path / "script.py"
    cachedir = tmp_path / "cache"
    script.write_text(
        "from os.path import basename\n"
        "from fscacher import PersistentCache\n"
        f"cache = PersistentCache(path={str(cachedir)!r})\n"
        "\n"
        "@cache.memoize_path\n"
        "def func(path):\n"
        "    print('Running %s.' % basename(path), end='')\n"
        "    return 'DONE'\n"
        "\n"
        f"print(func({str(script)!r}))\n"
    )

    outputs = [
        run([sys.executable, str(script)], stdout=PIPE, stderr=PIPE) for i in range(3)
    ]
    print("Full outputs: %s" % repr(outputs))
    if b"File name too long" in outputs[0].stderr:
        # must be running during conda build which blows up paths with
        # _placehold_ers
        pytest.skip("seems to be running on conda and hitting the limits")
    assert outputs[0].stdout.strip().decode() == "Running script.py.DONE"
    for o in outputs[1:]:
        assert o.stdout.strip().decode() == "DONE"


def test_memoize_path_tokens(tmp_path, cache, cache_tokens):
    calls = []

    @cache.memoize_path
    def memoread(path, arg, kwarg=None):
        calls.append(["cache", path, arg, kwarg])
        with open(path) as f:
            return f.read()

    @cache_tokens.memoize_path
    def memoread_tokens(path, arg, kwarg=None):
        calls.append(["cache_tokens", path, arg, kwarg])
        with open(path) as f:
            return f.read()

    def check_new_memoread(call, arg, content, expect_first=True, expect_new=False):
        ncalls = len(calls)
        assert call(path, arg) == content
        assert len(calls) == ncalls + int(expect_first)
        assert call(path, arg) == content
        assert len(calls) == ncalls + int(expect_first) + int(expect_new)

    path = str(tmp_path / "file.dat")

    with open(path, "w") as f:
        f.write("content")

    time.sleep(cache._min_dtime * 1.1)
    # They both are independent, so both will cause a new readout
    check_new_memoread(memoread, 0, "content")
    check_new_memoread(memoread_tokens, 0, "content")


@pytest.mark.parametrize(
    "fscacher_value,mycache_value,cleared,ignored",
    [
        ("clear", None, True, False),
        ("q", "clear", True, False),
        ("ignore", "clear", True, False),
        ("clear", "", False, False),
        ("clear", "q", False, False),
        ("ignore", None, False, True),
        ("q", "ignore", False, True),
        ("clear", "ignore", False, True),
        ("ignore", "", False, False),
        ("ignore", "q", False, False),
    ],
)
def test_cache_control_envvar(
    mocker, monkeypatch, fscacher_value, mycache_value, cleared, ignored, tmp_path
):
    if fscacher_value is not None:
        monkeypatch.setenv("FSCACHER_CACHE", fscacher_value)
    else:
        monkeypatch.delenv("FSCACHER_CACHE", raising=False)
    if mycache_value is not None:
        monkeypatch.setenv("MYCACHE_CONTROL", mycache_value)
    else:
        monkeypatch.delenv("MYCACHE_CONTROL", raising=False)
    clear_spy = mocker.spy(PersistentCache, "clear")
    c = PersistentCache(path=tmp_path, envvar="MYCACHE_CONTROL")
    assert clear_spy.called is cleared
    assert c._ignore_cache is ignored


@pytest.mark.skipif(shutil.which("git-annex") is None, reason="git annex required")
def test_follow_moved_symlink(cache, tmp_path):
    calls = []

    @cache.memoize_path
    def memoread(path):
        calls.append([path])
        with open(path) as f:
            return f.read()

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)

    content = "This is test text.\n"
    git("init")
    git("annex", "init")
    (tmp_path / "file.txt").write_text(content)
    git("annex", "add", "file.txt")
    git("commit", "-m", "Create file")
    assert op.islink(tmp_path / "file.txt")

    assert memoread(tmp_path / "file.txt") == content
    assert len(calls) == 1
    assert memoread(tmp_path / "file.txt") == content
    assert len(calls) == 1

    git("mv", "file.txt", "text.txt")
    git("commit", "-m", "Rename file")

    assert memoread(tmp_path / "text.txt") == content
    assert len(calls) == 1
    assert memoread(tmp_path / "text.txt") == content
    assert len(calls) == 1

    (tmp_path / "subdir").mkdir()
    git("mv", "text.txt", op.join("subdir", "text.txt"))
    git("commit", "-m", "Move file")

    assert memoread(tmp_path / "subdir" / "text.txt") == content
    assert len(calls) == 1
    assert memoread(tmp_path / "subdir" / "text.txt") == content
    assert len(calls) == 1


def test_memoize_path_nonpath_arg(cache, tmp_path):
    calls = []

    @cache.memoize_path
    def memoread(filepath, arg, kwarg=None):
        calls.append([filepath, arg, kwarg])
        with open(filepath) as f:
            return f.read()

    path = str(tmp_path / "file.dat")
    with open(path, "w") as f:
        f.write("content")

    time.sleep(cache._min_dtime * 1.1)

    ncalls = len(calls)
    assert memoread(path, 1) == "content"
    assert len(calls) == ncalls + 1
    assert memoread(arg=1, filepath=path) == "content"
    assert len(calls) == ncalls + 1


def test_dir_fingerprint_order_irrelevant(tmp_path):
    start = time.time()
    file1 = tmp_path / "apple.txt"
    file1.write_text("Apple\n")
    os.utime(file1, (start - 1, start - 1))
    file2 = tmp_path / "banana.txt"
    file2.write_text("This is test text.\n")
    os.utime(file2, (start - 2, start - 2))
    file3 = tmp_path / "coconut.txt"
    file3.write_text("Lorem ipsum dolor sit amet, consectetur adipisicing elit\n")
    os.utime(file3, (start - 3, start - 3))
    df_tuples = []
    for file_list in [
        [file1, file2, file3],
        [file3, file2, file1],
        [file2, file1, file3],
    ]:
        dprint = DirFingerprint()
        for f in file_list:
            fprint = FileFingerprint.from_stat(os.stat(f))
            dprint.add_file(f, fprint)
        df_tuples.append(dprint.to_tuple())
    for i in range(1, len(df_tuples)):
        assert df_tuples[0] == df_tuples[i]
