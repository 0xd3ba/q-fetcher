# qfetcher.py -- Implements the Q-Fetcher prefetcher

import os
import numpy as np
from tqdm import tqdm
from prefetcher.delta_q import DeltaQTable
from prefetcher.reward_tracker import RewardTrackerTable
from utils.output_writer import OutputWriter
from utils.signature_hash import SignatureHash


class QFetcher:
    """ A Q-Learning based data prefetcher for caches """
    def __init__(self,
                 signature_bits,            # Number of bits to represent the hashed signature
                 signature_shift,           # Number of bits by which the signature is shifted to form the new signature
                 reward_table_entries,      # Number of entries in the reward tracking table
                 entry_epoch,               # Number of steps before which the Q-table entry gets updated
                 alpha,                     # Alpha value in the update equation for Q-Learning
                 gamma,                     # The discounting factor to use
                 epsilon,                   # The probability of exploration (epsilon-greedy sampling)
                 page_size_bytes,           # Size of each page (in bytes)
                 cache_line_size_bytes,     # Size of each cache line (in bytes)
                 output_dir,                # The directory to place the output files
                 output_pred_file,          # Where to write the predictions
                 output_q_file              # Where to write the q-values of the predictions
                 ):
        self.signature_bits = signature_bits
        self.signature_shift = signature_shift
        self.reward_table_entries = reward_table_entries
        self.entry_epoch = entry_epoch
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.page_size_bytes = page_size_bytes
        self.cache_line_size_bytes = cache_line_size_bytes
        self.output_dir = output_dir
        self.output_pred_file = output_pred_file
        self.output_q_file = output_q_file

        self.cache_blks_per_page = page_size_bytes // cache_line_size_bytes

        self.output_writer = None
        self.delta_q_table = None
        self.reward_tracker_table = None
        self.signature_hasher = None

    def initialize(self):
        """ Initializes the prefetcher by setting up the necessary tables and stuff """
        self.output_writer = OutputWriter(self.output_dir, self.output_pred_file, self.output_q_file)
        self.delta_q_table = DeltaQTable(signature_bits=self.signature_bits,
                                         signature_shift=self.signature_shift,
                                         alpha=self.alpha,
                                         gamma=self.gamma,
                                         epsilon=self.epsilon,
                                         page_size_bytes=self.page_size_bytes,
                                         cache_line_size_bytes=self.cache_line_size_bytes)

        self.reward_tracker_table = RewardTrackerTable(self.reward_table_entries,
                                                       self.entry_epoch,
                                                       self.delta_q_table)

        self.signature_hasher = self.delta_q_table.get_signature_hasher()

    def start(self, ip_trace, load_trace):
        """ Starts the prefetcher (>_<)"""

        delta_signature = 0  # Initial value of the delta signature
        _, prev_load_tag, prev_cache_blk = load_trace[0]

        # Start from the second address in the address trace
        for curr_ip, curr_load_tuple in tqdm(zip(ip_trace[1:], load_trace[1:])):

            curr_load_addr, curr_load_tag, curr_cache_blk = curr_load_tuple     # Unpack the load address's tuple

            # Check if there is a page jump
            page_jumped = (curr_load_tag != prev_load_tag)
            delta = curr_cache_blk - prev_cache_blk

            # If there was a page change, we do not issue a prefetch request
            # But we update the delta. This will be required for next prefetch request
            # NOTE: Don't move signature calculation above here. The delta is not fixed until we check for a page
            #       change. So we need to avoid calculating the delta signature before it is finalized
            if page_jumped:
                delta += self.cache_blks_per_page
                delta_signature = self.signature_hasher.next_signature(delta_signature, delta)
                self.check_if_prefetched(curr_load_addr, delta_signature)
            else:
                delta_signature = self.signature_hasher.next_signature(delta_signature, delta)
                self.issue_prefetch(curr_ip, curr_load_addr, curr_cache_blk, delta_signature)

            prev_load_tag = curr_load_tag
            prev_cache_blk = curr_cache_blk

    def check_if_prefetched(self, load_addr, delta_signature):
        """ Checks if a particular load address that lead to a page change was prefetched previously """
        self.reward_tracker_table.check_n_give_reward(load_addr, delta_signature)

    def issue_prefetch(self, curr_ip, curr_load_addr, curr_cache_blk, delta_signature):
        """ Issues the specified number of prefetch requests """

        # For the time-being, issue only a single prefetch request
        # Do an epsilon-greedy sampling
        # TODO: Add an option in config.json to change the max. number of prefetches per load
        # TODO: Add a logger to log the issued prefetches and the Q-values of the prefetch requests

        delta_idx, delta_next, offset = self.delta_q_table.get_next_offset(delta_signature)
        pref_addr = curr_load_addr + offset

        invalid_prefetch = (delta_next + curr_cache_blk > self.delta_q_table.largest_delta) or \
                           (delta_next + curr_cache_blk < 0)

        # Insert the prefetch request into the reward tracker table
        self.reward_tracker_table.insert(pref_addr, delta_next, delta_signature, invalid_prefetch)

        # The address should not have the '0x' prefix which is present in hexadecimal notations
        pref_addr_hex = hex(pref_addr).split('0x')[-1]
        if not invalid_prefetch:
            self.output_writer.write(curr_ip, pref_addr_hex, self.delta_q_table.delta_q_table[delta_signature, delta_idx])

    def stop(self):
        """ Marks the end of prefetching process. Does the required cleanup """
        self.output_writer.close()
