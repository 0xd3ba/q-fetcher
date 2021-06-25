# replacement_policy.py -- Implements the classes for various replacement policies
#                          All the policies must inherit

import numpy as np


class ReplacementPolicyBase:
    """ The base class for the replacement policies """

    def find_victim(self, logical_ctrs, valid_bits):
        """
        logical_ctrs is a numpy array that contains only the logical counter values for each entry
        from whatever container holds the data. Based on this information find the entry that needs
        to be replaced and return its index.

        Same goes for valid_bits, which represents the list of valid entries
        """
        raise NotImplementedError

    def check_invalid_entry(self, valid_bits):
        """ Checks for invalid entries and returns the index if it exists """
        return np.any(valid_bits != 1)


class LRU(ReplacementPolicyBase):
    """ The class for least-recently used policy """

    def find_victim(self, logical_ctrs, valid_bits):
        """ Picks the victim with the least counter value """
        if self.check_invalid_entry(valid_bits):
            victim_idx = valid_bits.argmin()
        else:
            victim_idx = logical_ctrs.argmin()

        return victim_idx
