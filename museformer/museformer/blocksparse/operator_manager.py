import torch

from .optimized_matmul import matmul as BlocksparseMatMul
from .optimized_softmax import softmax as BlocksparseSoftmax
from ..tools.singleton_decorator import singleton


def layout_full_zero_check(input_layout):
    row_check = input_layout.sum(dim=2).eq(0)  # (head, row_len // block)
    col_check = input_layout.sum(dim=1).eq(0)  # (head, col_len // block)
    row_answer = bool(row_check.any())
    col_answer = bool(col_check.any())
    return row_answer or col_answer, row_answer, col_answer, row_check, col_check


class CacheList(object):
    def __init__(self, cache_size):
        self._cache = list()
        self._num = 0
        self.cache_size = cache_size

    def __len__(self):
        return self._num

    def __iter__(self):
        for idx in range(self._num - 1, -1, -1):
            yield self._cache[idx]

    def __getitem__(self, idx):
        return self._cache[idx]

    def put(self, x):
        if self._num >= self.cache_size:
            self._cache = self._cache[1:]
            self._num -= 1
        self._cache.append(x)
        self._num += 1

    def clear(self):
        self._cache.clear()
        self._num = 0


class OperatorManager(object):
    def __init__(self, operator_cls, cache_size=0):
        self.operator_cls = operator_cls
        self.cache_size = cache_size if cache_size > 0 else 0
        self.cache_dict = None if self.cache_size == 0 else {}

    def clear_cache(self):
        self.cache_dict.clear()

    def get_operator(self, layout, block, **kwargs):
        assert layout.dtype == torch.bool
        has_empty, row_answer, col_answer, _, _ = layout_full_zero_check(layout)
        assert not has_empty, "layout has empty %s, which may lead to error computation. Please check and fix." % (
            'row' if row_answer else 'column'
        )
        layout_cpu = None
        all_args = None
        if self.cache_dict is not None:
            layout_cpu = layout.cpu()
            all_args = (('block', block),) + tuple(kwargs.items())
            all_args = tuple(sorted(all_args, key=lambda x: x[0]))
            if all_args in self.cache_dict:
                cache_list = self.cache_dict[all_args]
                for cached_layout, operator in cache_list:
                    if torch.equal(layout_cpu, cached_layout):
                        return operator
        operator = self.operator_cls(layout=layout.long(), block=block, **kwargs)
        if self.cache_dict is not None:
            if all_args not in self.cache_dict:
                self.cache_dict[all_args] = CacheList(self.cache_size)
            self.cache_dict[all_args].put((layout_cpu, operator))
        return operator


@singleton
class BlocksparseMatMulManager(OperatorManager):
    def __init__(self, cache_size=0):
        super().__init__(BlocksparseMatMul, cache_size=cache_size)


@singleton
class BlocksparseSoftmaxManager(OperatorManager):
    def __init__(self, cache_size=0):
        super().__init__(BlocksparseSoftmax, cache_size=cache_size)
