# qfetcher.py -- Implements the Q-Fetcher prefetcher

import os
import numpy as np
from prefetcher.delta_q import DeltaQTable
from prefetcher.reward_tracker import RewardTrackerTable
from utils.output_writer import OutputWriter


class QFetcher:
    """ A Q-Learning based data prefetcher for caches """
    def __init__(self,
                 signature_bits,            # Number of bits to represent the hashed signature
                 reward_table_entries,      # Number of entries in the reward tracking table
                 entry_epoch,               # Number of steps before which the Q-table entry gets updated
                 alpha,                     # Alpha value in the update equation for Q-Learning
                 gamma,                     # The discounting factor to use
                 epsilon,                   # The probability of exploration (epsilon-greedy sampling)
                 page_size_bytes,           # Size of each page (in bytes)
                 cache_line_size_bytes,     # Size of each cache line (in bytes)
                 output_dir,                # The directory to place the output files
                 output_file                # Where to write the predictions
                 ):
        self.signature_bits = signature_bits
        self.reward_table_entries = reward_table_entries
        self.entry_epoch = entry_epoch
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.page_size_bytes = page_size_bytes
        self.cache_line_size_bytes = cache_line_size_bytes
        self.output_dir = output_dir
        self.output_file = output_file

        self.output_writer = None
        self.delta_q_table = None
        self.reward_tracker_table = None

    def initialize(self):
        """ Initializes the prefetcher by setting up the necessary tables and stuff """
        self.output_writer = OutputWriter(self.output_dir, self.output_file)
        self.delta_q_table = DeltaQTable(signature_bits=self.signature_bits,
                                         alpha=self.alpha,
                                         gamma=self.gamma,
                                         epsilon=self.epsilon,
                                         page_size_bytes=self.page_size_bytes,
                                         cache_line_size_bytes=self.cache_line_size_bytes)

        self.reward_tracker_table = RewardTrackerTable(self.reward_table_entries, self.entry_epoch)


    def start(self, ip_trace, load_trace):
        pass

    def stop(self):
        """ Marks the end of prefetching process. Does the required cleanup """
        self.output_writer.close()
