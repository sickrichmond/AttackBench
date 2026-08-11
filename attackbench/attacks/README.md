# Adding an attack

To register an attack you need to implement two functions.

### Config function

The first function declares the attack's parameters and should follow the template:

```python
def <library prefix>_<config name>():
    return dict(
        name='<name of the attack>',
        source='<name of the library>',
        threat_model='<name of the threat model>',
        option1=0.01,
    )
```

The library prefix corresponds to a shorter version of the library name; for instance, Foolbox's prefix is `fb`.
Every other key of the returned dict corresponds to one of the attack's options.
This function should be placed in the `<library>/configs.py` file.

> Historical note: config functions used to be written as a body of bare assignments,
> which the registry recovered by parsing the function's *source text*. That parser only
> understood plain literals, so values such as `1e-6`, `-1` or `1/30` were dropped
> silently and replaced by generic defaults — the attack then ran with parameters that
> were not the ones written in its config. Returning a dict removes the parser and the
> whole class of bug.

### Getter function

The second function to implement is the getter function, which will return the attack as a callable. This getter
function should follow the template:

```python
def get_<library prefix>_<attack name>(option1: float) -> Callable:
    return ...
```

The `<attack name>` in the name of the getter function should match exactly the name of the attack in the config
function: `name='<attack name>'`. This is necessary to determine which getter function to call when calling a named
config.

### Example

In `foolbox/configs.py`, the DDN attack is added with the two functions:

```python
def fb_ddn():
    return dict(
        name='ddn',
        source='foolbox',
        threat_model='l2',
        init_epsilon=1,
        num_steps=100,
        gamma=0.05,
    )


def get_fb_ddn(init_epsilon: float, num_steps: int, gamma: float) -> Callable:
    return partial(DDNAttack, init_epsilon=init_epsilon, steps=num_steps, gamma=gamma)
```

Additionally, one could add a second named config for DDN by simply implementing a second config function:

```python
def fb_ddn_large_gamma():
    return dict(
        name='ddn',
        source='foolbox',
        threat_model='l2',
        init_epsilon=1,
        num_steps=100,
        gamma=0.5,
    )
```

This second named config would point to the same getter function.
