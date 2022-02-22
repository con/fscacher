.. image:: https://github.com/con/fscacher/workflows/Test/badge.svg?branch=master
    :target: https://github.com/con/fscacher/actions?workflow=Test
    :alt: CI Status

.. image:: https://codecov.io/gh/con/fscacher/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/con/fscacher

.. image:: https://img.shields.io/pypi/pyversions/fscacher.svg
    :target: https://pypi.org/project/fscacher/

.. image:: https://img.shields.io/github/license/con/fscacher.svg
    :target: https://opensource.org/licenses/MIT
    :alt: MIT License

`GitHub <https://github.com/con/fscacher>`_
| `PyPI <https://pypi.org/project/fscacher/>`_
| `Issues <https://github.com/con/fscacher/issues>`_
| `Changelog <https://github.com/con/fscacher/blob/master/CHANGELOG.md>`_

``fscacher`` provides a cache & decorator for memoizing functions whose outputs
depend upon the contents of a file argument.

If you have a function ``foo()`` that takes a file path as its first argument,
and if the behavior of ``foo()`` is pure in the *contents* of the path and the
values of its other arguments, ``fscacher`` can help cache that function, like
so:

.. code:: python

    from fscacher import PersistentCache

    cache = PersistentCache("insert_name_for_cache_here")

    @cache.memoize_path
    def foo(path, ...):
        ...

Now the outputs of ``foo()`` will be cached for each set of input arguments and
for a "fingerprint" (timestamps & size) of each ``path``.  If ``foo()`` is
called twice with the same set of arguments, the result from the first call
will be reused for the second, unless the file pointed to by ``path`` changes,
in which case the function will be run again.

Caches are stored on-disk and thus persist between Python runs.  To clear a
given ``PersistentCache`` and erase its data store, call the ``clear()``
method.

By default, caches are stored in the user-wide cache directory, under an
fscacher-specific folder, with each one identified by the name passed to the
constructor (which defaults to "cache" if not specified).  To specify a
different location, use the ``path`` argument to the constructor instead of
passing a name:

.. code:: python

    cache = PersistentCache(path="/my/custom/location")

If your code runs in an environment where different sets of libraries or the
like could be used in different runs, and these make a difference to the output
of your function, you can make the caching take them into account by passing a
list of library version strings or other identifiers for the current run as the
``token`` argument to the ``PersistentCache`` constructor.

Finally, ``PersistentCache``'s constructor also optionally takes an ``envvar``
argument giving the name of an environment variable.  If that environment
variable is set to "``clear``" when the cache is constructed, the cache's
``clear()`` method will be called at the end of initialization.  If the
environment variable is set to "``ignore``" instead, then caching will be
disabled, and the cache's ``memoize_path`` method will be a no-op.  If the
given environment variable is not set, or if ``envvar`` is not specified, then
``PersistentCache`` will query the ``FSCACHER_CACHE`` environment variable
instead.


Installation
============
``fscacher`` requires Python 3.6 or higher.  Just use `pip
<https://pip.pypa.io>`_ for Python 3 (You have pip, right?) to install it and
its dependencies::

    python3 -m pip install fscacher
