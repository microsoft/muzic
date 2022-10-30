def add_submodule_args(cls, parser, submodules_attr_name='_submodules'):
    submodules = getattr(cls, submodules_attr_name, None)
    if submodules is None:
        return
    for submodule in submodules:
        if hasattr(submodule, 'add_args'):
            submodule.add_args(parser)


def str_bool(c):
    c_lower = c.lower()
    if c_lower in ('true', 'yes', 'y'):
        return True
    elif c_lower in ('false', 'no', 'n'):
        return False
    else:
        return None


def str_bool_with_default_error(c):
    r = str_bool(c)
    if r is None:
        raise ValueError('Value "%s" is not valid.' % c)
    else:
        return r


def str_to_type_with_specific_word_as_none(type_func, none_word):
    def f(x):
        return None if x == none_word else type_func(x)
    return f


def comma_split_tuple_of_specific_type(type_func):
    def inner(x):
        return tuple([type_func(item) for item in x.split(',')])
    return inner


def possibly_extend_tuple(c, n):
    assert isinstance(c, tuple)
    if len(c) == 1 and n > 1:
        c = c * n
    else:
        assert len(c) == n, \
            "%s for %d layers, len(c) == %d, type(c) == %s" % \
            (str(c), n, len(c), str(type(c)))
    return c
