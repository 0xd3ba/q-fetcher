# process_trace.py -- Preprocess the given load trace

import sys
import pathlib
import numpy as np
import pandas as pd


class PreprocessAddress:
    """
    Class responsible for preprocessing the given address by splitting it into two parts:
        1. Cache Block Number
        2. Rest of the address
    """
    def __init__(self, page_size, cache_line_size):
        self.page_size = page_size
        self.cache_line_size = cache_line_size

        self.offset_bits = int(np.log2(cache_line_size))
        self.page_bits = int(np.log2(page_size))
        self.block_bits = self.page_bits - self.offset_bits
        self.block_mask = int('0b' + ('1' * self.block_bits), base=2)

        # Some sanity checks
        assert self.block_bits > 0, f"Invalid cache line size: {cache_line_size}"
        assert self.page_bits > 0, f"Invalid page size: {page_size}"
        assert cache_line_size <= page_size, \
            f"Block size ({cache_line_size}) is greater than page size ({page_size}) "

    def preprocess(self, address):
        """ Splits the given address (as a hexadecimal string) """
        address_int = int(address, base=16)

        block_id = (address_int >> self.offset_bits) & self.block_mask  # Extract out the block ID
        tag_id = address_int >> self.page_bits  # Extract out the tag

        return address_int, tag_id, block_id


class PreprocessLoadTrace:
    """
    Class responsible for preprocessing the load trace
    (format as provided by ISCA 2021 ML-Prefetching competition)

    The columns of the trace file are as follows:
            instruction_id    cycle_count    load_address    ip_load    llc_hit_miss

    """
    def __init__(self, trace_dir, trace_file, page_size_bytes, cache_line_size_bytes):
        self.trace_file = trace_file
        self.trace_dir = trace_dir
        self.trace_file_path = pathlib.Path(trace_dir) / trace_file
        self.address_preprocess = PreprocessAddress(page_size_bytes, cache_line_size_bytes)

        # Doesn't matter what we name them, all we need is the dataframe column corresponding to
        # the target load/store address and the corresponding instruction pointer
        self.load_address = 'load_address'
        self.ip_address = 'instruction_id'

        self.columns = [self.ip_address,
                        'cycle_count',
                        self.load_address,
                        'load_ip',
                        'llc_hit_miss']

    def preprocess(self):
        """ Read and extract the load/store column """
        trace_df = None

        print(f'Trying to load trace file: {self.trace_file} ... ')

        try:
            trace_df = pd.read_csv(self.trace_file_path)
        except FileNotFoundError:
            print(f'Trace file "{self.trace_file}" does not exist')
            sys.exit(1)

        print('Trace file loaded successfully. Starting to preprocess it ... ')

        trace_df.columns = self.columns         # Need to add the columns because the trace file does not have them
        trace_load_seq = trace_df[self.load_address].apply(self.address_preprocess.preprocess).values
        trace_ip_seq = trace_df[self.ip_address].values

        print(f'Preprocessing done ... {trace_ip_seq.shape[0]} addresses in total.')
        # NOTE: The load address column has all the values as tuples of the form
        #                           (address, tag, block_id)
        return trace_ip_seq, trace_load_seq
