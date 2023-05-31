# Author: Botao Yu

def convert_basic_token_to_token_str(token):
    try:
        return '%s-%d' % (token[0], token[1])
    except TypeError:
        print(token)
        raise


def convert_basic_token_list_to_token_str_list(token_list):
    return [convert_basic_token_to_token_str(token) for token in token_list]


def convert_token_lists_to_token_str_lists(token_lists):
    str_lists = []

    for token_list in token_lists:
        str_list = convert_basic_token_list_to_token_str_list(token_list)
        str_lists.append(str_list)

    return str_lists


def convert_basic_token_str_to_token(token_str):
    try:
        t, value = token_str.split('-')
        value = int(value)
    except ValueError:
        print("Cannot convert the token_str to a valid_token:",  token_str)
        raise
    return t, value


def convert_basic_token_str_list_to_token_list(token_str_list):
    return [convert_basic_token_str_to_token(item) for item in token_str_list]
