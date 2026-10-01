# Custom fingerprints and git-annex content

Design notes for `memoize_path(custom_fingerprint=...)` and
`fscacher.annex_key_fingerprint`, introduced in
[#113](https://github.com/con/fscacher/pull/113).  The docstrings and README
only say what these are; this records why they work the way they do.

## Motivation

`memoize_path` keys its cache on a `stat()` of the path argument (mtime,
ctime, size, inode).  Some resources can vouch for their content better:

- A *locked* git-annex'ed file is a symlink whose target names a key; for
  content-hash backends, the key pins the content.  It also survives the file
  being moved, re-cloned, or its content dropped.
- Non-path objects (e.g. dandi-cli's `Readable`s, which may stream remote
  content) have no `stat()` at all, but may know a fingerprint of their
  content.

## `custom_fingerprint`: an alternative to `stat()`, not a second code path

The callable replaces only the `stat()`-based fingerprint; everything else
(falling back to a direct call when there is no fingerprint, injecting the
fingerprint into the cache key, tokens) is shared.  `_get_fingerprint()`
returns either a `CustomFingerprint` or a `PathFingerprint`, or `None`, and the
rest of `memoize_path` does not care which.  `PathFingerprint` keys exactly as
before, so existing caches stay valid.

The contract for a callable:

- **It returns `None` to fall back to `stat()`**, for anything it does not
  recognize, including plain paths unless it fingerprints them.  It must not
  raise for such values: an exception propagates, so the decorated call
  fails.  This is deliberate, as an exception there most likely is a bug in
  the callable, which should not silently disable caching.  (A way to say "do
  not cache", e.g. a named `fscacher.NO_CACHE` constant, can be added if ever
  needed.)
- **It runs on every call** of the decorated function, so it must be cheap.
  `annex_key_fingerprint` only reads the symlink; it never runs git-annex.
- **Equal fingerprints share results, wherever they are.**  The path argument
  itself is not part of the cache key: in `stat()` mode the (dereferenced) path
  is part of the fingerprint, but a custom fingerprint is used as is.  So a
  callable must include the path in its fingerprint unless the result does not
  depend on it, as `annex_key_fingerprint` does by default (see
  `pair_with_path` below).  For the same reason the value need not be a path
  at all.
- **Fingerprints must be picklable with a stable `repr()`**, as joblib hashes
  them into the key (e.g. a string or a tuple of strings).
- **There is no "modified just now" window.**  With `stat()`, a file modified
  within `_min_dtime` is not cached, since a quick later modification might
  not change the fingerprint.  A custom fingerprint is trusted to change
  whenever the result may change; the callable is responsible for that.  This
  is only justified for fingerprints that pin the content, like hash-based
  annex keys.

## Which git-annex files are fingerprinted by key

Only **locked** files: a symlink whose target looks like
`.../annex/objects/*/*/KEY/KEY`.  The link is read with `readlink()` only.

Only **content-hash backends**: `SHA1`, `SHA224`/`256`/`384`/`512`,
`SHA3_224`/`256`/`384`/`512`, `SKEIN256`/`512`, `BLAKE2B*`/`BP*`/`S*`/`SP*`,
and `MD5`, each with or without the `E` suffix (which adds the extension).
Everything else falls back to `stat()`:

- `WORM` keys are made of size, mtime and file name, so they do not pin the
  content.
- `URL` and `VURL` keys name a URL whose content may change.
- External `X*` backends are unknown.

**Unlocked files** also fall back to `stat()`: an unlocked file is a regular
file whose key is not updated until it is re-added, so an edit would not
change the key.  This includes every file on an *adjusted unlocked branch*,
which git-annex uses on filesystems without usable symlinks or permissions,
notably native Windows.  There, `annex_key_fingerprint` is a no-op.

WSL behaves like Linux on its own filesystem (e.g. under `~`).  On a Windows
drive under `/mnt/c`, git-annex is likely to treat the filesystem as crippled
and use an adjusted branch, so the same `stat()` fallback applies.  Neither is
tested in CI.

## `pair_with_path`

By default the fingerprint is `(absolute path, key)`: the path as given, not
dereferenced (dereferencing would give the object path, identical for all
files with the key).  Results are then only shared by files at the same path,
which is needed whenever the result depends on the path, e.g. on the extension
(the key's extension may differ from the file's) or on neighboring files.
Pairing is not *sufficient* for the latter, though: it keeps results separate
per location, but does not make them change when a neighbor does (see
[Other files the function reads](#other-files-the-function-reads)).
The path is made absolute with `os.path.abspath`, which does not resolve
symlinks (neither the file's nor its parent directories'), so the same file
reached through different paths gets separate entries; that errs on the safe
side.

With `pair_with_path=False`, the fingerprint is the key alone, and results are
shared by all files with the same content: twins in a dataset, and the same
file across clones.  Only use it for results that depend on the content only.

## Other files the function reads

**Only the first argument is fingerprinted**, in `stat()` mode and with a
custom fingerprint alike.  If the decorated function also reads other files,
e.g. the BIDS sidecar `sub-01_bold.json` of `sub-01_bold.nii.gz`, or metadata
inherited from parent directories, changes to those files do not change the
cache key, so the stale result is returned.  This predates custom
fingerprints and is not specific to git-annex; `pair_with_path` does not
change it, as the pair `(path, key of the .nii.gz)` stays the same when only
the sidecar changes.

To have such changes noticed, either:

- fold fingerprints of the relevant files into a custom fingerprint, e.g.
  `(annex_key_fingerprint(path), *(fingerprints of the sidecars))`, keeping in
  mind that the callable runs on every call and must stay cheap, and that it
  must change whenever the result may (no "modified just now" window applies);
  or
- have the caller pass something that identifies the content of those files
  (e.g. their keys or `stat()`s, or the parsed sidecar itself) as extra
  arguments of the decorated function, which are part of the cache key.

## Dropped content

With `stat()`, a locked file whose content was dropped is a broken symlink: it
cannot be fingerprinted, so the function is called directly every time.

With the key, it is still fingerprinted, so a result cached while the content
was present is returned after `git annex drop`.  This is useful (e.g. metadata
of files no longer present locally) and deliberate.  A file whose content was
never present has no cached result, so the function is called and fails as
before.

## Directories

`memoize_path` fingerprints a directory argument by walking it and combining
the `stat()` of every entry (this predates custom fingerprints).  With `stat()`
alone, a single dropped annexed file (a broken symlink) makes the whole
directory unfingerprintable, and the mtimes of annexed files feed the
"modified just now" window.

**The callable only sees the top-level argument.**  For a directory, it may
return a fingerprint of the whole tree itself; if it returns `None` (as
`annex_key_fingerprint` does), the directory is walked and `stat()`-ed exactly
as before, dropped files included.  This keeps a single call site and a single
kind of argument, so the callable stays a plain alternative to `stat()`.

An earlier iteration also consulted the callable for each entry met during the
walk (passing the `os.DirEntry`, so that `annex_key_fingerprint` could skip
regular files without a system call), so that trees with dropped annexed files
could be cached.  It was dropped: it made the callable part of the `stat()`
code path through a second, implicit contract, and it is better designed
together with the question below.

**Open question.**  The walk is O(number of entries) on every call.  For very
large trees (Zarrs with millions of chunks, possibly with dropped chunks) a
cheaper fingerprint of the whole tree would be needed, e.g. the git tree hash
of a committed directory, or per-entry fingerprints within the walk as above.
