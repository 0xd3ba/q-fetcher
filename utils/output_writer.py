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

    def __init__(self, output_dir, output_pred_file, output_q_file):
        self.output_pred_file = output_pred_file
        self.output_q_file = output_q_file

        self.output_pred_path = pathlib.Path(output_dir) / output_pred_file
        self.output_q_path = pathlib.Path(output_dir) / output_q_file

        # Create the directory if it does not exist yet
        # NOTE: The current working directory is set to the root directory of the project
        if output_dir not in os.listdir():
            print(f'Output directory {output_dir} does not exist. Creating one ... ')
            os.mkdir(output_dir)

        # IMPORTANT: This will overwrite the files if they already exist
        self.opened_output_file = open(self.output_pred_path, 'w')
        self.opened_q_val_file = open(self.output_q_path, 'w')

        # Create the buffers to store the things to write
        self.pred_buffer = []
        self.q_buffer = []

    def write(self, instr_id, prefetch_addr, q_value):
        """ Appends an entry into the output file buffers """
        self.pred_buffer.append((instr_id, prefetch_addr))
        self.q_buffer.append(q_value)

    def close(self):
        """ Marks the end of output generation by dumping the contents into the files and closing them """
        for instr_id, prefetch_addr in self.pred_buffer:
            self.opened_output_file.write(f'{instr_id.strip()} {prefetch_addr.strip()}\n')

        for q_val in self.q_buffer:
            self.opened_q_val_file.write(f'{q_val}\n')

        self.pred_buffer = []
        self.q_buffer = []

        self.opened_output_file.close()
        self.opened_q_val_file.close()