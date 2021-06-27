# main.py -- Starts the prefetcher

from prefetcher.qfetcher import QFetcher
from utils.config_parser import ConfigParser
from preprocess.process_trace import PreprocessLoadTrace

################################################
# Location of the config file. Set this to the
# path where the config.json file is located
CONFIG_JSON = './config.json'
################################################


if __name__ == '__main__':
    parser = ConfigParser(CONFIG_JSON)
    parser.parse()

    trace_preprocessor = PreprocessLoadTrace(**parser.get_trace_config(), **parser.get_system_config())
    prefetcher = QFetcher(**parser.get_q_fetcher_config(),
                          **parser.get_system_config(),
                          **parser.get_output_config())

    ip_trace, load_trace = trace_preprocessor.preprocess()

    prefetcher.initialize()
    prefetcher.start(ip_trace, load_trace)
    prefetcher.stop()