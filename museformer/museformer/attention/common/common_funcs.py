def print_redundant_params(kwargs, class_name=None):
    print('=====Redundant Params%s=====' % ('' if class_name is None else ' for %s' % class_name))
    for key in kwargs:
        print('{key}:\t{value}'.format(key=key, value=kwargs[key]))
    print('=============')
