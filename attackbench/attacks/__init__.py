minimal_search_steps = 20
minimal_init_eps = {
    'l0': 100,
    'l1': 10,
    'l2': 1,
    'linf': 1 / 255,
}

# Pre-configured attack instances — importable as `from attackbench.attacks import pgd`.
# Built lazily: importing this package must not require every attack library to be
# installed, since parts of it (bomn, the registry's optional libraries) need none.
_PRECONFIGURED = ('pgd', 'fgsm', 'apgd', 'fab', 'fmn', 'deepfool', 'superdeepfool',
                  'trust_region')


def __getattr__(name: str):
    if name in _PRECONFIGURED:
        from .. import preconfigured
        return getattr(preconfigured, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + list(_PRECONFIGURED)


__all__ = ['minimal_search_steps', 'minimal_init_eps', *_PRECONFIGURED]
