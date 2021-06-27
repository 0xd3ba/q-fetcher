# delta_q.py -- Module containing the class for the Delta Q-Table

# Number of offsets is twice the number of cache blocks (negative and positive offsets)
# For example if 4KB page and 64 byte blocks, then blocks per page = 4096/64 = 64
# and the offsets are [-127, 127]
# As for the reason,
# Within a page, the valid offsets are [-63, 63]
# There might be inter-page jumps. So in worst case, it might move from first block of first page
# to last block of next page or vive-versa:
#       - In forward direction:   0 -> 63 --page jump--> 0 --> 63, total offset = 63 + 1 + 63 = 127
#       - In backward direction: 63 -> 0  --page jump--> 63 --> 0, total offset = -63 -1 -63 = -127
#
# Hence, the number of offsets = (4 * n_blocks_per_page) - 1
# BUT, there is no point in trying to fetch a block at offsets not in the range [-63, 63] because we don't
# know the physical address of the next page, so the valid deltas are still [-63, 63]

import numpy as np
from utils.signature_hash import SignatureHash


class DeltaQTable:
    """

    Class implementing the Q-table for the deltas
    """

    def __init__(self,
                 signature_bits,
                 signature_shift,
                 alpha,
                 gamma,
                 epsilon,
                 page_size_bytes,
                 cache_line_size_bytes):

        self.signature_bits = signature_bits
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.page_size_bytes = page_size_bytes
        self.cache_line_size_bytes = cache_line_size_bytes
        self.delta_q_table = None

        # Some required sanity checks
        assert (page_size_bytes & (page_size_bytes-1)) == 0, \
            f"Page size must be a power of 2. Given {page_size_bytes} instead"

        assert (cache_line_size_bytes & (cache_line_size_bytes-1)) == 0, \
            f"Cache line size must be a power of 2. Given {cache_line_size_bytes} instead"

        assert signature_bits > 1, f"Signature bits must be > 1. Given {signature_bits} instead"

        self.n_cache_blks_per_page = page_size_bytes // cache_line_size_bytes
        self.n_cache_line_offsets = (2 * self.n_cache_blks_per_page) - 1
        self.n_delta_q_entries = 1 << signature_bits
        self._create()  # Create the table

        self.largest_delta = self.n_cache_blks_per_page - 1
        self.least_delta = -self.largest_delta    # Need this value for proper column indexing

        self.signature_hasher = SignatureHash(signature_bits, signature_shift, self.largest_delta)

    def _create(self):
        """ Creates the delta table based on the parameters """
        self.delta_q_table = np.zeros(shape=(self.n_delta_q_entries, self.n_cache_line_offsets))

    def _get_column_index(self, delta):
        """ Returns the index of the column for the given delta """
        return delta - self.least_delta

    def _get_delta_from_idx(self, col_idx):
        """ Returns the offset for the given column index """
        return col_idx + self.least_delta

    def get_next_offset(self, delta_signature):
        """ Returns the offset for the current signature """
        if np.random.uniform() < self.epsilon:
            # Alright, exploration time. The prefetch might be a terrible one, but exploration is needed !
            # NOTE: There is no need for a scheduler to decay the exploration rate over time. The reason being
            #       if the access pattern changes, then the Q-values will become invalid. So there is a need to start
            #       exploration yet again. Since there is no way of knowing when the pattern changes, keep the rate
            #       constant. Hence no need to decay it :)
            delta_idx = np.random.choice(range(self.delta_q_table.shape[1]))
        else:
            delta_idx = self.delta_q_table[delta_signature, :].argmax()

        delta = self._get_delta_from_idx(delta_idx)
        actual_offset = delta * self.cache_line_size_bytes
        return delta_idx, delta, actual_offset

    def get_signature_hasher(self):
        """ Returns a reference of the signature hashing method. Needed by the prefetcher """
        return self.signature_hasher

    def update(self, signature, delta, reward):
        """
        Responsible for updating the given entry (indexed using the signature) with the supplied values
        Applies Q-learning update rule:
            Q[S,A] = Q[S,A] + alpha*(reward + gamma* max_A(Q[S_next, A]) - Q[S,A])
        """

        s_curr = signature
        a_curr = self._get_column_index(delta)

        s_next = self.signature_hasher.next_signature(s_curr, delta)  # Index of the next entry
        a_next = self.delta_q_table[s_next].argmax()  # Q-learning requires max q-value of the action of next entry

        # Now apply the Q-Learning update rule on the current value
        self.delta_q_table[s_curr, a_curr] += self.alpha*(reward + self.gamma*self.delta_q_table[s_next, a_next] -
                                                          self.delta_q_table[s_curr, a_curr])
