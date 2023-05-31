def print_redundant_parameters(kwargs):
    if len(kwargs) > 0:
        print('The following parameters are redundant and do not function in the current situation:')
        for key in kwargs:
            print('%s\t%s' % (key, str(kwargs[key])))
