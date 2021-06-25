# config_parser.py -- Parser for parsing the config file located in root directory of the project

import json
import pathlib
from collections import OrderedDict


class ConfigParser:
    """ Parses the config file and stores the obtained information """

    def __init__(self, config_file):
        self.config_file = config_file
        self.config_file_path = pathlib.Path(config_file)

        # Following are the keys present in the JSON file
        # They must be the same. If new ones are added, make sure
        # to add them here
        self.trace_config_key = "trace_config"
        self.system_config_key = "system_config"
        self.q_fetcher_config_key = "q_fetcher_config"
        self.output_config_key = "output_config"

    def parse(self):
        config_file = None
        parsed_config = None

        print(f'Trying to parse {self.config_file} ... ')

        try:
            config_file = open(self.config_file_path, 'r')
        except FileNotFoundError:
            print(f'Config file ({self.config_file}) does not exist')
            exit(1)

        # Might face some issues in decoding the json file
        # due to invalid syntax or something
        try:
            parsed_config = json.load(config_file, object_hook=OrderedDict)
        except json.JSONDecodeError:
            print(f'Error parsing the config file ({self.config_file})')

        # Now save the obtained values as object variables
        for key, value in parsed_config.items():
            setattr(self, key, value)

        print(f'Parsing {self.config_file} successful ... ')

    def get_trace_config(self):
        return getattr(self, self.trace_config_key)

    def get_system_config(self):
        return getattr(self, self.system_config_key)

    def get_q_fetcher_config(self):
        return getattr(self, self.q_fetcher_config_key)

    def get_output_config(self):
        return getattr(self, self.output_config_key)