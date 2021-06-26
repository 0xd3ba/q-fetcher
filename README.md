# Q-fetcher

*Q-Fetcher* is an experimental prototype of a simple reinforcement-learning based hardware prefetcher
for CPU caches, which relies on Tabular Q-Learning to (hopefully) learn to prefetch particular cache blocks 
whose usage frequency would be the most in the near future. It does not require any pre-training, but instead learns
on the fly, i.e. online. 
It borrows ideas from [Signature Path Prefetcher (SPP)](https://ieeexplore.ieee.org/document/7783763)
and [Instruction Pointer Classifier-based Spatial Hardware Prefetching (IPCP)](https://ieeexplore.ieee.org/document/9138971)
to represent the sequence of *deltas* (difference between cache block # of consecutive accesses) as compressed
fixed-width *signatures*. 

**NOTE:** This was supposed to be a submission for 
[ML-Based Data Prefetching Competition (ISCA 2021)](https://sites.google.com/view/mlarchsys/isca-2021/ml-prefetching-competition?authuser=0)
but did not make it, due to time-constraints.

## Load Traces
The dataset consists of the LLC (*Last-Level Cache*) accesses (along with instruction IDs and other values) from 
several memory-intensive benchmarks by SPEC-2006, SPEC-2017 and GAP benchmark suites, which can be found on the competition
website [here](https://sites.google.com/view/mlarchsys/isca-2021/ml-prefetching-competition?authuser=0).

- Download any of the trace (`.xz` files inside `Load Traces/*/` directories), extract it and place the extracted `.txt` file inside `q-fetcher/traces/` directory.
- Inside `q-fetcher/config.json`, set the name of the trace file accordingly.

## Usage
(Recommended that a virtual environment is set up before proceeding further)

- Install the dependencies: `pip3 install requirement.txt`
- Place the appropriate trace file(s) into `q-fetcher/traces/` directory, as mentioned above
- (Optional) Change the parameters inside `q-fetcher/config.json` file related to the prefetcher
- Execute `q-fetcher/main.py` to start the prefetcher: `python3 main.py`

## Output
The output files will be stored to `q-fetcher/output/*.txt` and will replace previous files
(unless the name is changed inside `q-fetcher/config.json`). The output files consist of
1. Prefetch Addresses, in the format as specified in the competition's website.
2. Q-values of the corresponding prefetched addresses, just for the sake of debugging.

To evaluate the performance of the prefetcher, follow the instructions from 
[this repository](https://github.com/Quangmire/ChampSim).