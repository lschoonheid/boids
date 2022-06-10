"""
Python code for repeating boids simulations with different parameters to see
the effect of parameters on simulation outcome.

Execute: `python3 compare.py -h` for usage.

Student: Laszlo Schoonheid
Student ID: 11642610
Course: Programmeerproject 2022
"""


import argparse
from typing import Iterable
from matplotlib import pyplot as plt
import pandas
from tqdm import tqdm
from modules.helpers import pickle_cache, read_json
from repeat import repeat


@pickle_cache
def vary(
        key: str,
        var_range: Iterable,
        config_path: str = "config/repeat_config.json",
        config: dict = None,
        repeats: int = 1,
        **args
):
    """Execute simulations with variated configurations. Variates parameter `key` through `var_range`."""

    # Update config
    if not config:
        config = read_json(config_path)
    compare_config = {
        'repeats': repeats,
        'output': None,
        'config_path': None,
        'nofig': True,
        'config': config
    }

    data = dict()
    # Execute simulations while updating `key` value
    for var in tqdm(var_range):
        compare_config['config'][key] = var
        data[var] = repeat(**compare_config)

    return data


def plot_model_data(data, key, var_range, take_avg: bool = True, **kwargs):
    """Takes `data` and plots it with appropriate lay-out."""
    fig, ax = plt.subplots()
    # Set title to correspond with column name(s) of first model DataFrame in data
    xlabels = ", ".join([*data[var_range[0]][0][1].columns])
    plt.title(f'{xlabels} by variation of {key}')
    plt.xlabel('step')
    # Set ylabel to column name(s) of first model DataFrame in data
    plt.ylabel(xlabels)

    # Plot each instance or average data of each configuration
    for var in var_range:
        model_data = [instance[1] for instance in data[var]]
        if take_avg:
            model_data = [pandas.concat(model_data, axis=1).mean(axis=1)]
        for instance in model_data:
            plt.plot(instance, label=f"{key} = {var}")
    plt.legend()
    plt.show()
    return fig


if __name__ == '__main__':
    # Create a command line argument parser
    parser = argparse.ArgumentParser(description='Repeat boids simulation with variation of parameters')
    parser.add_argument("key", type=str, help="Key to direct parameter range to.")
    parser.add_argument('-r', nargs="+", type=int, dest='var_range', help="Parameters to try for `key`", required=True)
    parser.add_argument("-c", type=str, dest='config_path', default="config/default_config.json", help="Input base configuration file. Usage: -i config.json")
    parser.add_argument("-i", type=int, dest='repeats', default=1, help="Set number of times to repeat simulations per parameter.")
    parser.add_argument("-o", type=str, dest='output', default="outputs/compare/compare.png", help="Set number of times to repeat simulations per parameter.")
    parser.add_argument("--noavg", dest='take_avg', action='store_false', help="Generate plot without taking averages.")
    # Read arguments from command line
    args = vars(parser.parse_args())

    output = args.pop('output')

    data = vary(**args)
    plot = plot_model_data(data, **args)
    if output:
        plot.savefig(output)
