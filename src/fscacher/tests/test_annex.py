from functools import partial
import os
import os.path as op
from pathlib import Path
import shutil
import subprocess
import time
import pytest
from .. import PersistentCache, annex_key_fingerprint

KEY = "SHA256E-s7--0123456789abcdef.dat"


@pytest.fixture(scope="function")
def cache(tmp_path_factory):
    return PersistentCache(path=tmp_path_factory.mktemp("cache"))


def annex_link(repo: Path, relpath: str, key: str, content="content") -> Path:
    """
    Create a locked annexed file in a fake annex layout; with ``content=None``,
    its content is missing (as if dropped)
    """
    obj = repo / ".git" / "annex" / "objects" / "Xx" / "Yy" / key / key
    obj.parent.mkdir(parents=True, exist_ok=True)
    if content is not None:
        obj.write_text(content)
    link = repo / relpath
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(op.relpath(obj, link.parent), link)
    except OSError:
        pytest.skip("symlinks are not supported here")
    return link


def drop(link: Path) -> None:
    os.unlink(op.realpath(link))


@pytest.mark.parametrize(
    "key",
    [
        KEY,
        "SHA256-s7--0123456789abcdef",
        "SHA1-s7--0123",
        "SHA512E-s7--0123.nwb",
        "SHA3_256E-s7--0123.dat",
        "SKEIN512-s7--0123",
        "BLAKE2B256E-s7--0123.dat",
        "BLAKE2SP224-s7--0123",
        "MD5E-s7--0123.dat",
    ],
)
def test_annex_key_fingerprint_content_hash_backends(tmp_path, monkeypatch, key):
    link = annex_link(tmp_path, "sub/file.dat", key)
    assert annex_key_fingerprint(link) == (str(link), key)
    assert annex_key_fingerprint(str(link)) == (str(link), key)
    assert annex_key_fingerprint(os.fsencode(link)) == (str(link), key)
    assert annex_key_fingerprint(link, pair_with_path=False) == key
    # Relative paths are paired as absolute ones
    monkeypatch.chdir(tmp_path)
    assert annex_key_fingerprint(op.join("sub", "file.dat")) == (str(link), key)


@pytest.mark.parametrize(
    "key",
    [
        "WORM-s7-m1700000000--file.dat",
        "URL--https&c%%example.com%file.dat",
        "VURL--https&c%%example.com%file.dat",
        "XFOO-s7--0123",
        "SHA256Z-s7--0123",
    ],
)
def test_annex_key_fingerprint_other_backends(tmp_path, key):
    assert annex_key_fingerprint(annex_link(tmp_path, "file.dat", key)) is None


def test_annex_key_fingerprint_not_annexed(tmp_path):
    regular = tmp_path / "regular.dat"
    regular.write_text("content")
    other = tmp_path / "other.dat"
    try:
        # a link to a file named like a key, but not in an annex object store
        (tmp_path / KEY).mkdir()
        (tmp_path / KEY / KEY).write_text("content")
        os.symlink(op.join(KEY, KEY), other)
    except OSError:
        pytest.skip("symlinks are not supported here")
    for value in [regular, other, tmp_path, tmp_path / "missing", 42, None]:
        assert annex_key_fingerprint(value) is None


def test_annex_key_fingerprint_dropped(tmp_path):
    link = annex_link(tmp_path, "file.dat", KEY, content=None)
    assert not op.exists(link)
    assert annex_key_fingerprint(link) == (str(link), KEY)


def test_annex_key_fingerprint_dir_entries(tmp_path, monkeypatch):
    link = annex_link(tmp_path, "ds/file.dat", KEY)
    (tmp_path / "ds" / "regular.dat").write_text("content")
    readlinks = []
    readlink = os.readlink

    def spy(path):
        readlinks.append(os.fspath(path))
        return readlink(path)

    monkeypatch.setattr(os, "readlink", spy)
    with os.scandir(tmp_path / "ds") as entries:
        fprints = {e.name: annex_key_fingerprint(e) for e in entries}
    assert fprints == {"file.dat": (str(link), KEY), "regular.dat": None}
    # Only the symlink is read: a regular file is recognized from the listing
    assert readlinks == [str(link)]


def make_reader(cache, calls, **kwargs):
    @cache.memoize_path(**kwargs)
    def read(path):
        calls.append(str(path))
        with open(path) as f:
            return f"{op.basename(path)}:{f.read()}"

    return read


@pytest.mark.parametrize("pair_with_path", [True, False])
def test_memoize_path_annex_twins(cache, tmp_path, pair_with_path):
    calls = []
    read = make_reader(
        cache,
        calls,
        custom_fingerprint=partial(
            annex_key_fingerprint, pair_with_path=pair_with_path
        ),
    )
    # Twins: same key at different paths, with different extensions
    a = annex_link(tmp_path, "a.dat", KEY)
    b = annex_link(tmp_path, "sub/b.nwb", KEY)
    assert read(a) == "a.dat:content"
    assert read(a) == "a.dat:content"
    assert calls == [str(a)]
    if pair_with_path:
        # each path has its own entry, as the result may depend on it
        assert read(b) == "b.nwb:content"
        assert read(b) == "b.nwb:content"
        assert calls == [str(a), str(b)]
    else:
        # the result is shared, even though here it depends on the path
        assert read(b) == "a.dat:content"
        assert calls == [str(a)]


def test_memoize_path_annex_dropped(cache, tmp_path):
    calls = []
    read = make_reader(cache, calls, custom_fingerprint=annex_key_fingerprint)
    link = annex_link(tmp_path, "file.dat", KEY)
    assert read(link) == "file.dat:content"
    drop(link)
    # The result cached while the content was present is still returned
    assert read(link) == "file.dat:content"
    assert calls == [str(link)]
    # A file whose content was never present is read (and fails) every time
    missing = annex_link(tmp_path, "missing.dat", "SHA256E-s3--fedcba.dat", None)
    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            read(missing)
    assert calls == [str(link), str(missing), str(missing)]


def test_memoize_path_annex_fallback_to_stat(cache, tmp_path):
    calls = []
    read = make_reader(cache, calls, custom_fingerprint=annex_key_fingerprint)
    # A WORM key does not vouch for the content: stat() is used, which cannot
    # fingerprint a dropped file, so the function is called every time
    worm = annex_link(tmp_path, "worm.dat", "WORM-s7-m1--worm.dat", None)
    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            read(worm)
    assert calls == [str(worm)] * 2
    # An unlocked file (a regular file) is fingerprinted by stat(), so its
    # modifications are noticed
    calls.clear()
    unlocked = tmp_path / "unlocked.dat"
    unlocked.write_text("content")
    time.sleep(cache._min_dtime * 1.1)
    assert read(unlocked) == "unlocked.dat:content"
    assert read(unlocked) == "unlocked.dat:content"
    time.sleep(cache._min_dtime * 1.1)
    unlocked.write_text("edited")
    time.sleep(cache._min_dtime * 1.1)
    assert read(unlocked) == "unlocked.dat:edited"
    assert calls == [str(unlocked)] * 2


def test_memoize_path_annex_directory(cache, tmp_path):
    calls = []

    def listdir(path):
        calls.append(str(path))
        return sorted(
            (p.relative_to(path).as_posix(), op.exists(p))
            for p in Path(path).rglob("*")
            if not p.is_dir() or p.is_symlink()
        )

    with_keys = cache.memoize_path(listdir, custom_fingerprint=annex_key_fingerprint)
    with_stat = cache.memoize_path(listdir)
    ds = tmp_path / "ds"
    # the object store lives outside of the directory, as for a .zarr in a
    # dataset
    for name in ["a.dat", "sub/b.dat"]:
        annex_link(tmp_path, f"ds/{name}", f"SHA256E-s7--{name[-5]}.dat")
    drop(ds / "sub" / "b.dat")
    (ds / "regular.txt").write_text("regular")
    time.sleep(cache._min_dtime * 1.1)
    expected = [("a.dat", True), ("regular.txt", True), ("sub/b.dat", False)]
    # With stat() alone, the dropped file prevents fingerprinting the directory
    assert with_stat(ds) == with_stat(ds) == expected
    assert calls == [str(ds)] * 2
    # With keys, the directory is fingerprinted and its listing cached
    calls.clear()
    assert with_keys(ds) == with_keys(ds) == expected
    assert calls == [str(ds)]
    # ... until a file changes: an annexed one gets another key ...
    os.unlink(ds / "a.dat")
    annex_link(tmp_path, "ds/a.dat", "SHA256E-s6--other.dat")
    assert with_keys(ds) == expected
    assert len(calls) == 2
    # ... or a regular file is modified
    time.sleep(cache._min_dtime * 1.1)
    (ds / "regular.txt").write_text("modified")
    time.sleep(cache._min_dtime * 1.1)
    assert with_keys(ds) == with_keys(ds) == expected
    assert len(calls) == 3


@pytest.mark.skipif(shutil.which("git-annex") is None, reason="git annex required")
def test_memoize_path_git_annex(cache, tmp_path, monkeypatch):
    for var in ["GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"]:
        monkeypatch.setenv(var, "Test")
    for var in ["GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"]:
        monkeypatch.setenv(var, "test@example.com")

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)

    git("init", "-q")
    git("annex", "init", "-q")
    git("config", "annex.backend", "SHA256E")
    (tmp_path / "a.dat").write_text("content")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.dat").write_text("content")
    (tmp_path / "c.dat").write_text("other")
    (tmp_path / "w.dat").write_text("worm")
    git("annex", "add", "-q", "a.dat", "sub", "c.dat")
    git("-c", "annex.backend=WORM", "annex", "add", "-q", "w.dat")
    git("commit", "-q", "-m", "Add files")
    a, b, c, w = (tmp_path / p for p in ["a.dat", "sub/b.dat", "c.dat", "w.dat"])
    if not op.islink(a):
        pytest.skip("git-annex does not lock files here (adjusted branch?)")

    key = annex_key_fingerprint(a, pair_with_path=False)
    assert key.startswith("SHA256E-s7--")
    assert annex_key_fingerprint(b, pair_with_path=False) == key
    assert annex_key_fingerprint(w) is None

    calls = []
    read = make_reader(
        cache,
        calls,
        custom_fingerprint=partial(annex_key_fingerprint, pair_with_path=False),
    )
    assert read(a) == "a.dat:content"
    assert read(b) == "a.dat:content"
    assert calls == [str(a)]

    # Unlocked, a file is fingerprinted by stat(), and edits are noticed
    assert read(c) == "c.dat:other"
    git("annex", "unlock", "-q", "c.dat")
    assert annex_key_fingerprint(c) is None
    time.sleep(cache._min_dtime * 1.1)
    assert read(c) == "c.dat:other"
    time.sleep(cache._min_dtime * 1.1)
    c.write_text("edited")
    time.sleep(cache._min_dtime * 1.1)
    assert read(c) == "c.dat:edited"
    assert calls == [str(a), str(c), str(c), str(c)]

    # Dropped, a locked file's cached result is still returned
    git("annex", "drop", "-q", "--force", "a.dat")
    assert not op.exists(a)
    assert read(a) == "a.dat:content"
    assert len(calls) == 4
