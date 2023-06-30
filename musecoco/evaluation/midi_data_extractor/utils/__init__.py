from . import pos_process
from . import data
from . import similarity

from .pos_process import get_bar_num_from_sample_tgt_pos
# get_bar_num_from_sample_tgt_pos: Get the number of bars for a sample.

from .remigen_process import (
    remove_instrument, count_token_num, count_bar_num, get_bar_ranges,
    get_instrument_played, get_instrument_seq,
    sample_bars
)
# remove_instrument: Remove an instrument from the remigen sequence.
# count_token_num: Count the number of a specific token in a remigen sequence.
# count_bar_num: Count the number of bars, including the complete bars and a possible incomplete bar.
# get_instrument_played: Get the instrument tokens played in a remigen sequence.
# get_instrument_seq: Get sub-sequence of an instrument.
# sample_bars: Get a certain number of bars.

from .random import seed_everything
from .file_list import generate_file_list, read_file_list
