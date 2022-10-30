import torch
import torch.nn as nn

from ..tools.singleton_decorator import singleton


def generate_index_matrix(num_chunks, device='cpu'):
    chunk_arange = torch.arange(num_chunks, device=device)
    rows = chunk_arange.view(num_chunks, 1).expand(num_chunks, num_chunks)
    cols = chunk_arange.view(1, num_chunks).expand(num_chunks, num_chunks)
    index_matrix = torch.stack((rows, cols), dim=-1)  # (num_chunks, num_chunks, 2)
    return index_matrix


def generate_mask_template(num_chunks, device='cpu'):
    return torch.ones(num_chunks, num_chunks, dtype=torch.bool, device=device)


def generate_diagonal_indices(num_chunks, device='cpu'):
    diagonal = torch.arange(num_chunks, device=device)
    diagonal = torch.stack((diagonal, diagonal), dim=-1)
    return diagonal


def generate_mask(head_range_command, num_chunks, mask_template=None, device='cpu'):
    """
    Generate a selection mask of size (num_chunks, num_chunks) according to range_command. Selected ones are True.
    :param head_range_command:
    :param num_chunks:
    :param mask_template:
    :param device:
    :return:
    """
    if head_range_command is None:
        return None

    if mask_template is None:
        mask_template = generate_mask_template(num_chunks, device=device)
    else:
        w, h = mask_template.shape
        assert w == h == num_chunks
        mask_template = mask_template.to(device)

    mask = torch.zeros_like(mask_template)
    for command_item in head_range_command:
        if isinstance(command_item, int):
            command_item = (command_item, command_item + 1)

        begin, end = command_item
        if begin is None:
            begin = -num_chunks + 1
        if end is None:
            end = num_chunks

        if begin >= end:
            continue

        left = torch.triu(mask_template, begin)
        right = torch.tril(mask_template, end - 1)
        mask = mask | (left & right)

    return mask


@singleton
class BlockSelectionTemplateManager(nn.Module):
    """
    Generate mask according to some specific range c, and provide related management. Singleton design mode.
    """

    INIT_MAX_CHUNK = 1024
    _RESERVED_ATTR_NAMES = ('max_chunks', 'index_matrix', 'mask_template', 'diagonal_indices', 'mask')

    def __init__(self, init_max_chunk=None):
        super().__init__()
        if init_max_chunk is None:
            init_max_chunk = self.__class__.INIT_MAX_CHUNK
        self._max_chunk = init_max_chunk

        index_matrix = generate_index_matrix(self._max_chunk)
        self.register_buffer('_index_matrix', index_matrix, persistent=False)

        mask_template = generate_mask_template(self._max_chunk)
        self.register_buffer('_mask_template', mask_template, persistent=False)

        diagonal_indices = generate_diagonal_indices(self._max_chunk)
        self.register_buffer('_diagonal_indices', diagonal_indices, persistent=False)

        self.__range_commands_and_names = []

    def __update_index_matrix(self, num_chunks):
        new_index_matrix = generate_index_matrix(num_chunks, device=self._index_matrix.device)
        self._index_matrix = new_index_matrix

    def __update_mask_template(self, num_chunks):
        new_mask_template = generate_mask_template(num_chunks, device=self._mask_template.device)
        self._mask_template = new_mask_template

    def __update_diagonal_indices(self, num_chunks):
        new_diagonal_indices = generate_diagonal_indices(
            num_chunks, device=self._diagonal_indices.device
        )
        self._diagonal_indices = new_diagonal_indices

    def __update_masks(self, num_chunks):
        w, h = self._mask_template.shape
        assert w == h == num_chunks, "Please update mask_template first."
        for range_command, mask_name in self.__range_commands_and_names:
            setattr(self, mask_name,
                    generate_mask(range_command, num_chunks,
                                  mask_template=self._mask_template,
                                  device=self._mask_template.device))

    def update(self, num_chunks):
        if num_chunks <= self._max_chunk:
            return
        self.__update_index_matrix(num_chunks)
        self.__update_mask_template(num_chunks)
        self.__update_diagonal_indices(num_chunks)
        self.__update_masks(num_chunks)
        self._max_chunk = num_chunks

    @property
    def max_chunk(self):
        return self._max_chunk

    @max_chunk.setter
    def max_chunk(self, num_chunks):
        self.update(num_chunks)

    @property
    def index_matrix(self):
        return self._index_matrix

    @property
    def mask_template(self):
        return self._mask_template

    @property
    def diagonal_indices(self):
        return self._diagonal_indices

    @property
    def device(self):
        return getattr(self, '_mask_template').device

    def register_range_command(self, range_command):
        name = str(range_command)
        if name in self.__class__._RESERVED_ATTR_NAMES:
            raise ValueError
        if hasattr(self, name):
            return
        mask = generate_mask(range_command, self.max_chunk, mask_template=self.mask_template)
        self.register_buffer(name, mask, persistent=False)
        self.__range_commands_and_names.append((range_command, name))

    def mask(self, range_command):
        name = str(range_command)
        assert (range_command, name) in self.__range_commands_and_names
        return getattr(self, name)

    def get_diagonal_indices(self, num_chunks):
        assert num_chunks > 0
        self.update(num_chunks)
        return self._diagonal_indices[:num_chunks]
