# 0.4.3 (Tue Nov 19 2024)

#### 🐛 Bug Fix

- Address lint warnings and drop Python 3.8 [#99](https://github.com/con/fscacher/pull/99) ([@yarikoptic](https://github.com/yarikoptic))

#### Authors: 1

- Yaroslav Halchenko ([@yarikoptic](https://github.com/yarikoptic))

---

# 0.4.2 (Tue Nov 19 2024)

#### 🧪 Tests

- Log when file timestamp unexpectedly in the future + more information in the failing test_memoize_path_dir [#98](https://github.com/con/fscacher/pull/98) ([@yarikoptic](https://github.com/yarikoptic))
- Stop testing against PyPy 3.8; retry failed `test_memoize_path_dir` on Windows [#95](https://github.com/con/fscacher/pull/95) ([@jwodder](https://github.com/jwodder))

#### Authors: 2

- John T. Wodder II ([@jwodder](https://github.com/jwodder))
- Yaroslav Halchenko ([@yarikoptic](https://github.com/yarikoptic))

---

# 0.4.1 (Tue Jun 04 2024)

#### 🐛 Bug Fix

- Stop using/testing EOLed 3.6 and 3.7, use 3.9 for linting (3.8 EOLs soon) [#91](https://github.com/con/fscacher/pull/91) ([@yarikoptic](https://github.com/yarikoptic) [@jwodder](https://github.com/jwodder))
- ASV dropped --strict option  in 0.6.0 release [#83](https://github.com/con/fscacher/pull/83) ([@yarikoptic](https://github.com/yarikoptic))

#### 🏠 Internal

- Add a few folders I found locally into git ignore [#92](https://github.com/con/fscacher/pull/92) ([@yarikoptic](https://github.com/yarikoptic))
- [gh-actions](deps): Bump codecov/codecov-action from 3 to 4 [#89](https://github.com/con/fscacher/pull/89) ([@dependabot[bot]](https://github.com/dependabot[bot]) [@jwodder](https://github.com/jwodder))
- [gh-actions](deps): Bump actions/setup-python from 4 to 5 [#88](https://github.com/con/fscacher/pull/88) ([@dependabot[bot]](https://github.com/dependabot[bot]))
- [gh-actions](deps): Bump actions/checkout from 3 to 4 [#82](https://github.com/con/fscacher/pull/82) ([@dependabot[bot]](https://github.com/dependabot[bot]))

#### 🧪 Tests

- Test against Python 3.12 and PyPy 3.10 [#84](https://github.com/con/fscacher/pull/84) ([@jwodder](https://github.com/jwodder))
- Use Python 3.8 to test against dev version of joblib [#86](https://github.com/con/fscacher/pull/86) ([@jwodder](https://github.com/jwodder))

#### 🔩 Dependency Updates

- Replace appdirs with platformdirs [#85](https://github.com/con/fscacher/pull/85) ([@jwodder](https://github.com/jwodder))

#### Authors: 3

- [@dependabot[bot]](https://github.com/dependabot[bot])
- John T. Wodder II ([@jwodder](https://github.com/jwodder))
- Yaroslav Halchenko ([@yarikoptic](https://github.com/yarikoptic))

---

# 0.4.0 (Wed Aug 16 2023)

#### 🚀 Enhancement

- Add `exclude_kwargs` to memoization decorators [#38](https://github.com/con/fscacher/pull/38) ([@yarikoptic](https://github.com/yarikoptic) [@jwodder](https://github.com/jwodder))

#### Authors: 2

- John T. Wodder II ([@jwodder](https://github.com/jwodder))
- Yaroslav Halchenko ([@yarikoptic](https://github.com/yarikoptic))

---

# 0.3.0 (Mon Feb 20 2023)

#### 🚀 Enhancement

- Ignore cache for non-path-like arguments [#79](https://github.com/con/fscacher/pull/79) ([@jwodder](https://github.com/jwodder))

#### 🐛 Bug Fix

- Drop support for Python 3.6 [#80](https://github.com/con/fscacher/pull/80) ([@jwodder](https://github.com/jwodder) [@yarikoptic](https://github.com/yarikoptic))

#### 🏠 Internal

- Update GitHub Actions action versions [#77](https://github.com/con/fscacher/pull/77) ([@jwodder](https://github.com/jwodder))

#### 🧪 Tests

- Test against more recent versions of PyPy [#81](https://github.com/con/fscacher/pull/81) ([@jwodder](https://github.com/jwodder))
- Test against Python 3.11 [#78](https://github.com/con/fscacher/pull/78) ([@jwodder](https://github.com/jwodder))
- Clean out vfat mount between benchmarks [#76](https://github.com/con/fscacher/pull/76) ([@jwodder](https://github.com/jwodder))

#### Authors: 2

- John T. Wodder II ([@jwodder](https://github.com/jwodder))
- Yaroslav Halchenko ([@yarikoptic](https://github.com/yarikoptic))

---

# 0.2.0 (Tue Feb 22 2022)

#### 🚀 Enhancement

- Support specifying a custom path for the cache; tokens becomes kwonly [#73](https://github.com/con/fscacher/pull/73) ([@jwodder](https://github.com/jwodder))
- make joblib ignore "path" , pass resolved as part of the fingerprinting kwargs arg [#63](https://github.com/con/fscacher/pull/63) ([@yarikoptic](https://github.com/yarikoptic) [@jwodder](https://github.com/jwodder))

#### 🏎 Performance

- Cache directory fingerprint as a XORed hash of file fingerprints [#71](https://github.com/con/fscacher/pull/71) ([@jwodder](https://github.com/jwodder))
- Don't fingerprint paths when caching is ignored [#72](https://github.com/con/fscacher/pull/72) ([@jwodder](https://github.com/jwodder))

#### 🏠 Internal

- Improve linting configuration [#64](https://github.com/con/fscacher/pull/64) ([@jwodder](https://github.com/jwodder))
- Make versioneer.py use setuptools instead of distutils [#54](https://github.com/con/fscacher/pull/54) ([@jwodder](https://github.com/jwodder))
- Update codecov action to v2 [#53](https://github.com/con/fscacher/pull/53) ([@jwodder](https://github.com/jwodder))

#### 🧪 Tests

- Make benchmarks measure cache misses and hits separately [#74](https://github.com/con/fscacher/pull/74) ([@jwodder](https://github.com/jwodder))
- Update Python version used to test development joblib to 3.7 [#65](https://github.com/con/fscacher/pull/65) ([@jwodder](https://github.com/jwodder))
- Capture all logs during tests [#56](https://github.com/con/fscacher/pull/56) ([@jwodder](https://github.com/jwodder))

#### Authors: 2

- John T. Wodder II ([@jwodder](https://github.com/jwodder))
- Yaroslav Halchenko ([@yarikoptic](https://github.com/yarikoptic))

---

# 0.1.6 (Thu Oct 07 2021)

#### 🐛 Bug Fix

- Revert "Limit joblib version to pre-1.1.0" [#52](https://github.com/con/fscacher/pull/52) ([@jwodder](https://github.com/jwodder))

#### 🧪 Tests

- Test against Python 3.10 [#49](https://github.com/con/fscacher/pull/49) ([@jwodder](https://github.com/jwodder))
- Change pypy3 to pypy-3.7 on GitHub Actions [#50](https://github.com/con/fscacher/pull/50) ([@jwodder](https://github.com/jwodder))

#### Authors: 1

- John T. Wodder II ([@jwodder](https://github.com/jwodder))

---

# 0.1.5 (Thu Oct 07 2021)

#### 🐛 Bug Fix

- Limit joblib version to pre-1.1.0 [#48](https://github.com/con/fscacher/pull/48) ([@jwodder](https://github.com/jwodder))
- Test against and update for dev version of joblib [#42](https://github.com/con/fscacher/pull/42) ([@jwodder](https://github.com/jwodder))

#### 🏠 Internal

- Resimplify release workflow [#35](https://github.com/con/fscacher/pull/35) ([@jwodder](https://github.com/jwodder))
- Remove debug step [#34](https://github.com/con/fscacher/pull/34) ([@jwodder](https://github.com/jwodder))

#### 🧪 Tests

- Test handling of moving symlinks around in git-annex [#47](https://github.com/con/fscacher/pull/47) ([@jwodder](https://github.com/jwodder))

#### Authors: 1

- John T. Wodder II ([@jwodder](https://github.com/jwodder))

---

# 0.1.4 (Mon Feb 22 2021)

#### 🐛 Bug Fix

- Fix versioneer+auto integration (or else) [#33](https://github.com/con/fscacher/pull/33) ([@jwodder](https://github.com/jwodder))

#### Authors: 1

- John T. Wodder II ([@jwodder](https://github.com/jwodder))

---

# 0.1.3 (Mon Feb 22 2021)

#### 🐛 Bug Fix

- Try to debug versioneer failure [#32](https://github.com/con/fscacher/pull/32) ([@jwodder](https://github.com/jwodder))

#### Authors: 1

- John T. Wodder II ([@jwodder](https://github.com/jwodder))

---

# 0.1.2 (Mon Feb 22 2021)

#### 🐛 Bug Fix

- Get auto and versioneer to play nice together [#31](https://github.com/con/fscacher/pull/31) ([@jwodder](https://github.com/jwodder))

#### Authors: 1

- John T. Wodder II ([@jwodder](https://github.com/jwodder))

---

# 0.1.1 (Mon Feb 22 2021)

#### 🐛 Bug Fix

- Get tests to pass on Windows and macOS [#29](https://github.com/con/fscacher/pull/29) ([@jwodder](https://github.com/jwodder))

#### ⚠️ Pushed to `master`

- Start CHANGELOG ([@jwodder](https://github.com/jwodder))

#### 🏠 Internal

- Set up auto [#27](https://github.com/con/fscacher/pull/27) ([@jwodder](https://github.com/jwodder))

#### 🧪 Tests

- Get asv to run pypy3 correctly [#30](https://github.com/con/fscacher/pull/30) ([@jwodder](https://github.com/jwodder))

#### Authors: 1

- John T. Wodder II ([@jwodder](https://github.com/jwodder))

---

# v0.1.0 (2021-02-18)

Initial release
