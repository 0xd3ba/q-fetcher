# process_trace.py -- Preprocess the given load trace

import pathlib
import numpy as np
import pandas as pd


class PreprocessAddress:
    """
    Class responsible for preprocessing the given address by splitting it into two parts:
        1. Cache Block Number
        2. Rest of the address
    """
    def __init__(self, cache_line_size):
        self.cache_line_size = cache_line_size

    def preprocess(self, address):
        """ Splits the given address (as a hexadecimal string) """
        block_bits = int(np.log2(self.cache_line_size))

        assert block_bits > 0, f"Invalid cache line size: {self.cache_line_size}"

        address_int = int(address, base=16)
        block_mask = int('0b' + ('1' * block_bits), base=2)

        block_id = address_int & block_mask
        tag_id = address_int >> block_bits

        return tag_id, block_id


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
        self.address_preprocess = PreprocessAddress(cache_line_size_bytes)

        # Doesn't matter what we name them, all we need is the dataframe column corresponding to
        # the target load/store address and the corresponding instruction pointer
        self.load_address = 'load_address'
        self.ip_address = 'ip_load'

        self.columns = ['instruction_id',
                        'cycle_count',
                        self.load_address,
                        self.ip_address,
                        'llc_hit_miss']

    def preprocess(self):
        """ Read and extract the load/store column """
        trace_df = None
        try:
            trace_df = pd.read_csv(self.trace_file_path)
        except FileNotFoundError:
            print(f'Trace file "{self.trace_file}" does not exist')
            exit(1)

        trace_df.columns = self.columns         # Need to add the columns because the trace file does not have them
        trace_load_seq = trace_df[self.load_address].apply(self.address_preprocess.preprocess).values
        trace_ip_seq = trace_df[self.ip_address].values

        # NOTE: The load address column has all the values as tuples of the form
        #                           (tag, block_id)
        return trace_load_seq, trace_ip_seq
