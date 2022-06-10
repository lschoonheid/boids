"""
Interface for a single simulation of boids.

Execute: `python3 main.py -h` for usage.

Student: Laszlo Schoonheid
Student ID: 11642610
Course: Programmeerproject 2022
"""


import argparse
from typing import Iterable
from modules.helpers import Animation, BoidModel, collect, read_json


def main(
        n_boids: int = 30,
        x: int = 20,
        y: int = 20,
        z: int = 20,
        speed: float = 0.1,
        i_max: int = 30,
        process_id: int = None,
        do_cache: bool = True,
        do_animation: bool = True,
        realtime: bool = False,
        show_anim: bool = True,
        anim_path: str = None,
        data_path: str = None,
        obstacles: Iterable = None,
        **kwargs: dict
):
    """Interface for executing boids simulation and optional animation generation."""

    # Test for correct input
    if show_anim and anim_path:
        print("WARNING: displaying animation and saving animation concurrently causes blitting errors.")
    assert not (data_path and realtime), "Saving data is only possible when prerendering animation."

    # Define additional model parameters from input
    # Only add 3rd coordinate if it has a value
    box_size = [coord for coord in (x, y, z) if coord]
    assert len(box_size) >= 2, "Box must have at least two dimensions."
    model_args = n_boids, box_size, speed
    model_kwargs = {'obstacles': obstacles}
    data = None

    # (pre)Render data if required
    if data_path or not realtime or not do_animation:
        if do_cache:
            data = collect(BoidModel, model_args, model_kwargs, i=i_max, process_id=process_id)
        else:
            data = collect.__wrapped__(BoidModel, model_args, model_kwargs, i=i_max, process_id=process_id)

        # Optionally save data to CSV
        if data_path:
            data.to_csv(data_path)

    if do_animation:
        # Render data with animation in realtime
        if realtime:
            anim = Animation(realtime=True, model=BoidModel(*model_args, **model_kwargs), i_max=i_max)
        # Render data, then show animation
        else:
            anim = Animation(data=data, box_size=box_size, config=model_kwargs)

        if show_anim:
            anim.show()

        # Optionally save animation as GIF
        if anim_path:
            anim.save(anim_path)

    if data is not None:
        return data


if __name__ == "__main__":
    # Create a command line argument parser
    parser = argparse.ArgumentParser(description='Simulate boids flying')
    parser.add_argument("-n", type=int, default=30, dest='n_boids', help="number of boids")
    parser.add_argument("-x", type=int, default=20, dest='x', help="x-dimension of box")
    parser.add_argument("-y", type=int, default=20, dest='y', help="y-dimension of box")
    parser.add_argument("-z", type=int, default=20, dest='z', help="z-dimension of box")
    parser.add_argument("-i", type=int, default=30, dest='i_max', help="Number of iterations")
    parser.add_argument("--noanim", dest='do_animation', action='store_false', help="Simulate without generating animation")
    parser.add_argument("--noshow", dest='show_anim', action='store_false', help="Simulate without showing animation")
    parser.add_argument("--realtime", action='store_true', help="Simulate in realtime")
    parser.add_argument("--savefig", dest='anim_path', help="Save animation to GIF. Usage: --savefig NAME.gif")
    parser.add_argument("--savedata", dest='data_path', help="Save data to CSV. Usage: --savedata NAME.csv")
    parser.add_argument("--nocache", dest='do_cache', action='store_false', help="Run simulation without caching output data")
    parser.add_argument("--config", dest='config_path', help="Configuration file for simulation.")

    # Read arguments from command line
    args = parser.parse_args()
    kwargs = vars(args)

    # If config file is provided, update kwargs with config file (overrides commandline arguments)
    config = kwargs.get('config_path', None)
    if config:
        kwargs.update(read_json(config))

    # Run simulation through interface with provided arguments
    main(**kwargs)
