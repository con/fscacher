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
