# output_writer.py -- Module containing the class for generating the output

import os
import pathlib


class OutputWriter:
    """
    Class responsible for creating the directories (if it doesn't exist) and saving the
    predictions in the format specified by ISCA 2021 ML Prefetching Competition:

                    Instruction_ID     Target_Prefetch_Address

    NOTE: Both should be hexadecimal strings
    """

    def __init__(self, output_dir, output_file):
        self.output_file = output_file
        self.output_path = pathlib.Path(output_dir) / output_file

        # Create the directory if it does not exist yet
        # NOTE: The current working directory is set to the root directory of the project
        if output_dir not in os.listdir:
            os.mkdir(output_dir)

        # IMPORTANT: This will overwrite the file if it already exists
        self.opened_output_file = open(self.output_path, 'w')

    def write(self, instr_id, prefetch_addr):
        """ Appends an entry into the output file """
        self.opened_output_file.write(f'{instr_id} {prefetch_addr}')

    def close(self):
        """ Marks the end of output generation by closing the file """
        self.opened_output_file.close()