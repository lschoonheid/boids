"""
Python code for repeating simulations with constant configuration,
enabling data mining and seeing average results.

Execute: `python3 repeat.py -h` for usage.

Student: Laszlo Schoonheid
Student ID: 11642610
Course: Programmeerproject 2022
"""


from copy import copy
import multiprocessing
from typing import Iterable
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from modules.helpers import hashargs, read_json
from main import main
import argparse


def wrapper(kwargs: dict):
    """Execute `main(**kwargs)` with `kwargs`."""
    return main(**kwargs)


def plot_counts(data: Iterable, output: str = None):
    """Takes `data` and plots flock counts."""
    flock_counts = [instance[1] for instance in data]
    fig, ax = plt.subplots()
    for counts in flock_counts:
        counts.plot(ax=ax)
    if output:
        plt.savefig(output)
    plt.close()


def repeat(
        output: str = None,
        repeats: int = 1,
        config_path: str = None,
        make_fig: bool = True,
        **kwargs: dict
):
    """Repeat simulation with given configuration for `repeats` times and return data."""

    # Update configuration
    if not config_path:
        config_path = "config/repeat_config.json"
    config_dict = kwargs.get('config', read_json(config_path))
    args_identifier = hashargs(**config_dict)
    # Generate iterable configurations
    configs = list()
    for i in range(repeats):
        config_dict['process_id'] = i
        configs.append(copy(config_dict))

    # Take advantage of multithreaded systems by mapping processes to different threads
    num_workers = multiprocessing.cpu_count()   
    pool = multiprocessing.Pool(num_workers)
    with pool as p:
        # Execute simulations with `configs`
        data = list(tqdm(p.imap(wrapper, configs), total=len(configs), desc="Instances", position=0, leave=" "))

    # Optionally save data to csv
    if output:
        pandata = pd.DataFrame(data)
        pandata.to_csv(output)

    # Optionally plot data to file.
    if make_fig:
        plot_counts(data, output=f"outputs/repeat_{repeats}_{args_identifier}.png")

    return data


if __name__ == "__main__":
    # Create a command line argument parser
    parser = argparse.ArgumentParser(description='Repeat boids simulation with variation of parameters')
    parser.add_argument("-i", type=int, dest='repeats', default=1, help="Number of times to repeat simulation. Usage: -i [i]")
    parser.add_argument("-o", type=str, dest='output', help="Output csv file. Usage: -o output.csv")
    parser.add_argument("-c", type=str, dest='config_path', help="Input configuration file. Usage: -i config.json")
    parser.add_argument("--nofig", dest='make_fig', action='store_false', help="Simulate without generating figure")
    # Read arguments from command line
    args = vars(parser.parse_args())

    repeat(**args)
