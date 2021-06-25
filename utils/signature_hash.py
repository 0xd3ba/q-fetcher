# signature_hash.py -- Module containing the class to hash the signatures

class SignatureHash:
    """
    Class responsible for generating "signatures" from a delta sequence.
    """
    def __init__(self, signature_bits, n_shifts, max_delta_val):
        self.signature_bits = signature_bits        # Number of bits to represent a hashed signature
        self.n_shifts = n_shifts                    # Number of bits to shift before XOR-ing

        # The maximum number of bits to store the delta in a sign-magnitude manner
        # For example, bin(127) = "0b1111111"
        # Splitting with '0b' will give ["", "1111111"]
        # Take the last part and find the length will give the number of bits
        # Add 1 to store the sign-magnitude
        self.max_bits_per_delta = len(bin(max_delta_val).split('0b')[-1]) + 1

    def _to_sign_magnitude(self, delta):
        """
        Converts the given delta to sign-magnitude format with (self.max_bits_per_delta + 1) bits
        The MSB corresponds to the sign of the value: 0 means positive, 1 means negative
        """
        bin_delta = bin(delta).split('0b')[-1]
        if delta < 0:
            bin_delta = '1' + bin_delta
        else:
            bin_delta = '0' + bin_delta

        return int(bin_delta, base=2)

    def next_signature(self, curr_signature, delta):
        """ Generates the next signature and returns it """
        sign_mag_delta = self._to_sign_magnitude(delta)
        new_signature = (curr_signature << self.n_shifts) ^ sign_mag_delta

        # Keep only the last "signature_bits" of the signature
        new_signature = new_signature & ((1 << self.signature_bits) - 1)
        return new_signature
