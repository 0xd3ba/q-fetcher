# reward_tracker.py -- Module containing the class for Reward-Tracking table
# The table is of the following format:
#
#         Prefetched_Address    Delta    Delta_signature    Reward    Steps    Timestamp    Valid_bit
#
#         - Prefetched_Address: An address whose prefetch request was issued previously
#         - Delta:              The offset with which the load address was added to generate the prefetched address
#         - Delta_signature:    The signature of the pattern with which the delta was chosen
#         - Reward:             The current reward issued
#         - Steps:              The number of steps this entry had been present in this table
#         - TimeStamp:          The last time when this entry was accessed. Used for LRU replacement
#         - Valid_bit:          Indicates if this entry is valid
#
#     The table can be implemented as a numpy matrix with the given number of entries and above mentioned columns
#
#     For looking up the table whether the current address was prefetched, we only need to check for
#     an entry with the matching address and the delta signature. Note that the same address might have been
#     prefetched with a different delta signature, in this case, it would be wrong to penalize this. So give it
#     a pseudo-hit (positive) reward which is less than the actual reward given for a hit.

import numpy as np
from utils.replacement_policy import LRU

################################################
# Rewards to be issued for a hit in this table
# and miss in this table
REWARD_HIT = 16
REWARD_PSEUDO_HIT = 8
REWARD_MISS = -1

# Columns in the reward table (see the class
# description below). In case a new column needs
# to be inserted, write down its name and insert
# into the list below
PREF_ADDRESS_COL = 'prefetched_address'
DELTA_COL = 'delta'
DELTA_SIG_COL = 'delta_signature'
REWARD_COL = 'reward'
STEP_COL = 'step'
TIMESTAMP_COL = 'timestamp'
VALID_BIT_COL = 'valid'

REWARD_TABLE_COLUMNS_LIST = [
    PREF_ADDRESS_COL,
    DELTA_COL,
    DELTA_SIG_COL,
    REWARD_COL,
    STEP_COL,
    TIMESTAMP_COL,
    VALID_BIT_COL
]
################################################


class RewardTrackerTable:
    """
    Reward tracking table to issue rewards to the issued prefetches and later used to update
    the corresponding Q-values of the deltas.
    """

    def __init__(self,
                 n_entries,
                 steps_per_entry,
                 delta_q_table,
                 hit_reward=REWARD_HIT,
                 semi_hit_reward=REWARD_PSEUDO_HIT,
                 miss_reward=REWARD_MISS):

        self.n_entries = n_entries
        self.steps_per_entry = steps_per_entry
        self.delta_q_table = delta_q_table
        self.reward_hit = hit_reward
        self.reward_miss = miss_reward
        self.reward_semi_hit = semi_hit_reward
        self.columns_list = REWARD_TABLE_COLUMNS_LIST
        self.logical_clock = 0

        # Number the columns appropriately and save them as instance variables
        for col_idx, column in enumerate(self.columns_list):
            setattr(self, column, col_idx)

        # The replacement policy used to find the entries that need to be replaced
        # If this needs to be changed, go to utils/replacement_policy.py and implement a new class
        # by following the instructions there and store an instance of that class below instead
        self.replacement_policy = LRU()

        self.reward_table = None
        self._create()  # Create the table and fill it with zeros

    def _create(self):
        """ Creates the table appropriately """
        self.reward_table = np.zeros(shape=(self.n_entries, len(self.columns_list)),
                                     dtype=np.int64)

        self._pref_addr_idx = getattr(self, PREF_ADDRESS_COL)   # Get the index for the prefetch address column
        self._step_idx = getattr(self, STEP_COL)                # Get the index for the step column
        self._valid_bit_idx = getattr(self, VALID_BIT_COL)      # Get the index of the valid bit address column
        self._delta_sig_idx = getattr(self, DELTA_SIG_COL)      # Get the index of the delta signature column
        self._delta_idx = getattr(self, DELTA_COL)              # Get the index of the delta column
        self._reward_idx = getattr(self, REWARD_COL)            # Get the index of the reward column
        self._time_idx = getattr(self, TIMESTAMP_COL)           # Get the index of the timestamp column

    def _lookup(self, load_addr, delta_signature):
        """
        Looks up the table for the appropriate entry and returns true if found, else false.
        Before returning, it also gives the appropriate rewards to the entries that got a hit and
        the remaining entries
        """

        # Look if there is a match and whether it is valid
        pref_matches = (self.reward_table[:, self._pref_addr_idx] == load_addr)
        delta_sig_matches = (self.reward_table[:, self._delta_sig_idx] == delta_signature)

        valid_entries = (self.reward_table[:, self._valid_bit_idx] > 0)  # Select all those entries with valid bit set

        # NOTE: The matches can be 0, 1 or >1
        # The first case, when this address was not prefetched.
        # The second case arises when the address was prefetched using *ANY* delta signature
        # The third case arises when the address was prefetched using different signatures and is currently present here
        entry_matches = np.logical_and(pref_matches, valid_entries)
        entry_and_delta_sig_matches = np.logical_and(entry_matches, delta_sig_matches)

        entry_found = (entry_matches.sum() > 0)

        return entry_found, entry_matches, entry_and_delta_sig_matches

    def _increment_ticks(self):
        """ Increments the logical clock for each entry """
        self.logical_clock += 1                     # Increment the timer by 1
        
    def _increment_steps(self):
        """ Increases the steps of all entries by 1 """
        self.reward_table[:, self._step_idx] += 1

    def _update_existing(self, idx_mask):
        """ Updates an existing entry by updating the timestamp """
        self.reward_table[idx_mask, self._time_idx] = self.logical_clock
        self.reward_table[idx_mask, self._reward_idx] += self.reward_hit

    def _invalidate_entries(self):
        """
        Invalidates the entries (ones who crossed the total steps) by updating their corresponding delta_q table
        and then setting the valid bit in the reward tracking table to 0. If there are no such entries, does
        nothing
        """

        valid_entries_ids_mask = (self.reward_table[:, self._valid_bit_idx] > 0)
        invalid_entries_ids_mask = (self.reward_table[:, self._step_idx] >= self.steps_per_entry)

        # Only care about the entries that have their valid bit set
        invalid_entries_ids_mask = invalid_entries_ids_mask * valid_entries_ids_mask

        # If there are no such entries, just return
        if invalid_entries_ids_mask.sum() == 0:
            return

        # There exists at least one invalid entries. Process each of them separately
        invalid_entries = self.reward_table[invalid_entries_ids_mask]
        for entry in invalid_entries:
            delta_sig = entry[self._delta_sig_idx]
            delta = entry[self._delta_idx]
            reward = entry[self._reward_idx]

            # Update the corresponding entries in the deltaQ-table
            self.delta_q_table.update(delta_sig, delta, reward)

        # Finally reset the valid bit
        self.reward_table[invalid_entries_ids_mask, self._valid_bit_idx] = 0

    def _issue_rewards(self, entry_found, all_matches, all_with_delta_sig_matches):
        """ Issues rewards accordingly """
        all_valid_entries = (self.reward_table[:, self._valid_bit_idx] > 0)
        self.reward_table[all_valid_entries, self._reward_idx] += self.reward_miss  # Penalize all valid entries
        self.reward_table[all_matches, self._reward_idx] -= self.reward_miss  # Cancel out penalty on those that matched

        # Get only the matches that match in addresses, but do not match delta signatures
        all_matches = np.logical_xor(all_matches, all_with_delta_sig_matches)

        if entry_found:
            if all_with_delta_sig_matches.sum() > 0:
                self._update_existing(all_with_delta_sig_matches)

            self.reward_table[all_matches, self._reward_idx] += self.reward_semi_hit
            # TODO: Do we need to update the timestamp for these entries ?

    def check_n_give_reward(self, load_addr, delta_signature):
        """
        Checks if a load address has been already prefetched previously
        NOTE: Only to be used when there is a page-change. This does not insert a new entry
        """
        entry_found, all_matches, delta_sig_matches = self._lookup(load_addr, delta_signature)

        # If the entry was not found, then it would be wrong to penalize all other entries because
        # we do not know if cache blocks from this page was previously fetched or not (well, didn't implement that ._.)
        # So no need for penalizing (for now ;) )
        if not entry_found:
            return

        self._issue_rewards(entry_found, all_matches, delta_sig_matches)


    def insert(self, pref_addr, delta, delta_signature):
        """
        Inserts an entry into the table. There are two cases that can happen
            1. There is already an entry with matching delta and delta signature
            2. There is no such entry. Need to replace one by LRU policy
        """
        self._invalidate_entries()  # Invalidate entries, if any
        self._increment_steps()     # Increment the steps

        addr_found, entry_matches_mask, entry_and_delta_sig_matches_mask = self._lookup(pref_addr, delta_signature)

        assert entry_and_delta_sig_matches_mask.sum() <= 1, f"ERROR: Multiple matches for {pref_addr} {delta_signature} found"

        # Replacement will be done, only when necessary. Else this variable remains unused
        victim_entry_idx = self.replacement_policy.find_victim(self.reward_table[:, self._time_idx],
                                                               self.reward_table[:, self._valid_bit_idx])

        self._issue_rewards(addr_found, entry_matches_mask, entry_and_delta_sig_matches_mask)

        # If we didn't find a perfect match, we need to allocate an entry and insert it
        # The victim entry needs to be written back to the delta-Q table, if it is a valid entry
        if entry_and_delta_sig_matches_mask.sum() == 0:
            self.delta_q_table.update(self.reward_table[victim_entry_idx, self._delta_sig_idx],
                                      self.reward_table[victim_entry_idx, self._delta_idx],
                                      self.reward_table[victim_entry_idx, self._reward_idx])
            # Finally, replace the entry
            self.reward_table[victim_entry_idx, self._pref_addr_idx] = pref_addr
            self.reward_table[victim_entry_idx, self._delta_idx] = delta
            self.reward_table[victim_entry_idx, self._delta_sig_idx] = delta_signature
            self.reward_table[victim_entry_idx, self._step_idx] = 0
            self.reward_table[victim_entry_idx, self._time_idx] = self.logical_clock
            self.reward_table[victim_entry_idx, self._valid_bit_idx] = 1
            self.reward_table[victim_entry_idx, self._reward_idx] = self.reward_hit

        self._increment_ticks()     # Increment the logical clock
