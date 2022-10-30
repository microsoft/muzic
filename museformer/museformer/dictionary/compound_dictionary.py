import re

from fairseq.data import Dictionary


class CompoundDictionary(Dictionary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_dict = None
        self.special_pad_word = None
        self.special_pad_index = None

    @classmethod
    def load(cls, f):
        d = cls()
        d.add_from_file(f)
        d.construct_types()
        return d

    def construct_types(self):
        type_dict = {}
        type_token = None
        current_begin = None
        considered_type_token = set()
        for idx, symbol in enumerate(self.symbols):
            match = re.fullmatch('([a-z]+)\-\d+', symbol)
            if match is None:
                if current_begin is not None:
                    type_dict[type_token] = (current_begin, idx)
                type_token = None
                current_begin = None
            else:
                now_type_token = match.group(1)
                if current_begin is not None:
                    if type_token == now_type_token:
                        continue
                    else:
                        type_dict[type_token] = (current_begin, idx)
                        assert now_type_token not in considered_type_token
                        type_token = now_type_token
                        considered_type_token.add(type_token)
                        current_begin = idx
                else:
                    assert now_type_token not in considered_type_token
                    type_token = now_type_token
                    considered_type_token.add(type_token)
                    current_begin = idx
        if current_begin is not None:
            type_dict[type_token] = (current_begin, len(self.symbols))
        self.type_dict = type_dict
        return type_dict
