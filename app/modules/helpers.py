"""
Includes all helper functions for running boids simulations.

Student: Laszlo Schoonheid
Student ID: 11642610
Course: Programmeerproject 2022
"""


from collections import namedtuple
from copy import copy
from functools import wraps
import hashlib
import json
import pickle
from random import uniform
from typing import Callable, Iterable, List, Tuple, Union
from matplotlib import animation, patches, pyplot as plt
from mesa import Model, Agent
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace, GridContent
from mesa.time import SimultaneousActivation
import numpy as np
from tqdm import tqdm


def dump_pickle(data, output: str):
    """Dump pickle file."""
    with open(output, 'wb') as handle:
        pickle.dump(data, handle)
    return data


def load_pickle(location: str):
    """Load pickle file."""
    with open(location, 'rb') as handle:
        data = pickle.load(handle)
    return data


def hashargs(*args, **kwds):
    """Takes `args` and `kwds` as arguments and hashes its information to a string."""
    args_identifier = hashlib.md5(str((args, kwds)).encode()).hexdigest()
    return args_identifier


def pickle_cache(func: Callable, quiet: bool = False):
    """Decorator function for caching function output to PKL files."""
    @wraps(func)
    def wrapper(*args, **kwds):
        args_identifier = hashargs(*args, **kwds)
        output = ".cache/" + args_identifier + ".pkl"

        try:
            data = load_pickle(output)
            if not quiet:
                print("Found cached data. Loading from cache instead.")
        except FileNotFoundError:
            data = dump_pickle(func(*args, **kwds), output)
        return data
    return wrapper


@pickle_cache
def collect(model: Model, model_args, model_kwargs, i: int = 1000, process_id: int = None):
    """Run simulation and collect data. Returns `DataFrame` object."""
    model = model(*model_args, **model_kwargs)

    # tqdm visualisation variables
    process_text = lambda i: f" {i}" if i is not None else ""
    position = lambda i: 1 + i if i is not None else None
    for i in tqdm(range(i),
                  desc=f"Rendering simulation{process_text(process_id)}",
                  position=position(process_id),
                  leave=" "):
        model.step()

    # Collect and return data
    agent_data = model.datacollector.get_agent_vars_dataframe()
    model_data = model.datacollector.get_model_vars_dataframe()
    return agent_data, model_data


def min_norm(*scalars: Iterable):
    """Returns the value with the smallest norm, but keeping its sign."""
    signs = np.sign(scalars)
    absolutes = np.abs(scalars)
    index, minimum = np.argmin(absolutes), min(absolutes)
    return minimum * signs[index]


def normalize(vector: Iterable[float]):
    """Takes `Iterable[float]`. Returns normalized vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def vector_avg(vectors: Iterable[Iterable[float]]):
    """Returns the mean position of `vectors`."""
    mean = np.zeros(len(vectors[0]))
    for v in vectors:
        mean += v
    mean /= len(vectors)
    return mean


def rotate_vector(vector: Iterable[float] = None, theta: float = None):
    """Rotate 2D vector."""
    assert len(vector) == 2, "Only 2D rotation is supported."
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rotation_matrix.dot(vector)


def random_vector(dimensions: int):
    """
    Generate random vector of length `dimensions`.

    Example: v = [x_1, x_2, ..., x_n] with x_n in [0, 1).
    """
    return np.array([uniform(0, 1) for _ in range(dimensions)])


FloatCoordinate = Union[Tuple[float, float, float], np.ndarray]


class ContinuousSpacePlus(ContinuousSpace):
    """
    Copy of Mesa ContinuousSpace class with options for toroidal space in spatial methods.

    Continuous space where each agent can have an arbitrary position.

    Assumes that all agents are point objects, and have a pos property storing
    their position as an (x, y, z) tuple. This class uses a numpy array internally
    to store agent objects, to speed up neighborhood lookups.
    """

    def _do_torus(self, torus_self, torus_arg):
        return ((torus_self and torus_arg is None) or torus_arg) and torus_arg is not False

    def get_neighbors(
        self, pos: FloatCoordinate, radius: float, include_center: bool = True, torus: bool = None
    ) -> List[GridContent]:
        """Get all objects within a certain radius.

        Args:
            pos: (x,y) coordinate tuple to center the search at.
            radius: Get all the objects within this distance of the center.
            include_center: If True, include an object at the *exact* provided
                            coordinates. i.e. if you are searching for the
                            neighbors of a given agent, True will include that
                            agent in the results.
        """
        deltas = np.abs(self._agent_points - np.array(pos))
        if self._do_torus(self.torus, torus):
            deltas = np.minimum(deltas, self.size - deltas)
        dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2

        (idxs,) = np.where(dists <= radius ** 2)
        neighbors = [
            self._index_to_agent[x] for x in idxs if include_center or dists[x] > 0
        ]
        return neighbors

    def get_distance(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate, torus: bool = None) -> float:
        """Get the distance between two point, accounting for toroidal space.

        Args:
            pos_1, pos_2: Coordinate tuples for both points.
        """
        x1, y1 = pos_1
        x2, y2 = pos_2

        dx = np.abs(x1 - x2)
        dy = np.abs(y1 - y2)
        if self._do_torus(self.torus, torus):
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return np.sqrt(dx * dx + dy * dy)

    def get_distance_vector(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate, torus: bool = None) -> float:
        """Get the distance vector between two point, accounting for toroidal space.

        Args:
            pos_1, pos_2: Coordinate tuples for both points.
        """
        x1, y1 = pos_1
        x2, y2 = pos_2

        dx = x1 - x2
        dy = y1 - y2
        if self._do_torus(self.torus, torus):
            dx = min_norm(dx, self.width - dx)
            dy = min_norm(dy, self.height - dy)
        return np.array([dx, dy])


class ContinuousSpace3D(ContinuousSpacePlus):
    """
    Copy of Mesa ContinuousSpace class with 3D expansion and options for
    toroidal space in spatial methods.

    Continuous space where each agent can have an arbitrary position.

    Assumes that all agents are point objects, and have a pos property storing
    their position as an (x, y, z) tuple. This class uses a numpy array internally
    to store agent objects, to speed up neighborhood lookups.
    """

    _grid = None

    def __init__(
        self,
        x_max: float,
        y_max: float,
        z_max: float,
        torus: bool,
        x_min: float = 0,
        y_min: float = 0,
        z_min: float = 0,
    ) -> None:
        """Create a new continuous space.

        Args:
            x_max, y_max, z_max: Maximum x, y and z coordinates for the space.
            torus: Boolean for whether the edges loop around.
            x_min, y_min, z_min: (default 0) If provided, set the minimum x, y and z
                          coordinates for the space. Below them, values loop to
                          the other edge (if torus=True) or raise an exception.
        """
        super().__init__(x_max, y_max, torus, x_min, y_min)

        self.z_min = z_min
        self.z_max = z_max
        self.depth = z_max - z_min
        self.center = np.array(((x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2))
        self.size = np.array((self.width, self.height, self.depth))

    def move_agent(self, agent: Agent, pos: FloatCoordinate) -> None:
        """Move an agent from its current position to a new position.

        Args:
            agent: The agent object to move.
            pos: Coordinate tuple to move the agent to.
        """
        pos = self.torus_adj(pos)
        idx = self._agent_to_index[agent]
        self._agent_points[idx, 0] = pos[0]
        self._agent_points[idx, 1] = pos[1]
        self._agent_points[idx, 2] = pos[2]
        agent.pos = pos

    def get_neighbors(self, pos: FloatCoordinate, radius: float,
                      include_center: bool = True, torus: bool = None
                      ) -> List[GridContent]:
        """Get all objects within a certain radius.

        Args:
            pos: (x,y,z) coordinate tuple to center the search at.
            radius: Get all the objects within this distance of the center.
            include_center: If True, include an object at the *exact* provided
                            coordinates. i.e. if you are searching for the
                            neighbors of a given agent, True will include that
                            agent in the results.
        """
        deltas = np.abs(self._agent_points - np.array(pos))
        if self._do_torus(self.torus, torus):
            deltas = np.minimum(deltas, self.size - deltas)
        dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2 + deltas[:, 2] ** 2

        (idxs,) = np.where(dists <= radius ** 2)
        neighbors = [
            self._index_to_agent[x] for x in idxs if include_center or dists[x] > 0
        ]
        return neighbors

    def get_distance(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate, torus: bool = None) -> float:
        """Get the distance between two point, accounting for toroidal space.

        Args:
            pos_1, pos_2: Coordinate tuples for both points.
        """
        x1, y1, z1 = pos_1
        x2, y2, z2 = pos_2

        dx = np.abs(x1 - x2)
        dy = np.abs(y1 - y2)
        dz = np.abs(z1 - z2)
        if self._do_torus(self.torus, torus):
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
            dz = min(dz, self.depth - dz)
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def get_distance_vector(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate, torus: bool = None) -> float:
        """Get the distance vector between two point, accounting for toroidal space.

        Args:
            pos_1, pos_2: Coordinate tuples for both points.
        """
        x1, y1, z1 = pos_1
        x2, y2, z2 = pos_2

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        if self._do_torus(self.torus, torus):
            dx = min_norm(dx, self.width - dx)
            dy = min_norm(dy, self.height - dy)
            dz = min_norm(dz, self.depth - dz)
        return np.array([dx, dy, dz])

    def torus_adj(self, pos: FloatCoordinate) -> FloatCoordinate:
        """Adjust coordinates to handle torus looping.

        If the coordinate is out-of-bounds and the space is toroidal, return
        the corresponding point within the space. If the space is not toroidal,
        raise an exception.

        Args:
            pos: Coordinate tuple to convert.
        """
        if not self.out_of_bounds(pos):
            return pos
        elif not self.torus:
            raise Exception("Point out of bounds, and space non-toroidal.")
        else:
            x = self.x_min + ((pos[0] - self.x_min) % self.width)
            y = self.y_min + ((pos[1] - self.y_min) % self.height)
            z = self.z_min + ((pos[2] - self.z_min) % self.depth)

            if isinstance(pos, tuple):
                return (x, y, z)
            else:
                return np.array((x, y, z))

    def out_of_bounds(self, pos: FloatCoordinate) -> bool:
        """Check if a point is out of bounds."""
        x, y, z = pos
        return x < self.x_min or x >= self.x_max or y < self.y_min or y >= self.y_max or z < self.z_min or z >= self.z_max


class Flock:
    """Class for flocks."""
    def __init__(self, agent: Agent = None):
        self.positions = list()
        if agent:
            self.positions.append(agent.pos)

    def add(self, agent: Agent):
        self.positions.append(agent.pos)


def count_flocks(model: Model):
    """Takes `Model` object and counts the amount of flocks present."""
    lenience = 3  # Distancing lenience for counting a boid towards a flock
    flocking_threshold = lenience * model.separation
    # Define the first flock as enveloping  the first boid
    boids = filter_type(model.schedule.agents, Boid)
    flocks = [Flock(boids[0])]
    # Iterate over each boid and verify whether it's close enough to a flock
    # to be a part of it
    for free_agent in boids[1:]:
        # Abort current loop if `free_agent` has been added to a flock
        is_flocked = False
        for flock in flocks:
            if not is_flocked:
                for flocked_pos in flock.positions:
                    distance = model.space.get_distance(flocked_pos, free_agent.pos)
                    if distance < flocking_threshold:
                        flock.add(free_agent)
                        is_flocked = True
                        break
        # Create new flock if boid is not close enough to any flock
        if not is_flocked:
            flocks.append(Flock(free_agent))
    return len(flocks)


class Obstacle(Agent):
    """Base class for obstacles."""
    def __init__(self, unique_id: int, model: Model) -> None:
        """Create new obstacle."""
        super().__init__(unique_id, model)
        self.dir = None


class Rectangle(Obstacle):
    """Class for rectangle obstacle."""
    def __init__(
        self,
        unique_id: int,
        model: Model,
        width: float = None,
        height: float = None,
        center: Iterable[float] = [0, 0],
        rotation: float = 0
    ) -> None:
        super().__init__(unique_id, model)

        """Create new obstacle."""
        self.width = width
        self.height = height
        self.center = center
        self.rotation = rotation

    def hit(self, x: float, y: float):
        """Check if `x` and `y` coordinates lay within obstacle region."""
        # Transform coordinates to vector to apply linear algebra
        v = np.array([x, y])
        # Apply inverse transformation
        v -= self.center
        if self.rotation != 0:
            v = rotate_vector(v, -self.rotation)
        # Obtain adjusted coordinates
        x, y = v
        if -self.width <= 2 * x <= self.width and -self.height <= 2 * y <= self.height:
            return True
        else:
            return False


def filter_type(_list: list, _type: type, include: bool = True):
    """Filter list on `_type`"""
    filtered = list()
    for item in _list:
        if include:
            if type(item) == _type:
                filtered.append(item)
        else:
            if type(item) != _type:
                filtered.append(item)
    return filtered


class Boid(Agent):
    """Class for a single boid."""

    def __init__(
        self,
        unique_id: int,
        model: Model,
        pos: np.ndarray,
        dir: np.ndarray,
        vision: float = 5,
        separation: float = 1
    ) -> None:
        """Create new boid."""
        super().__init__(unique_id, model)
        self.pos = pos
        self.dir = dir
        self.dim = len(pos)
        self.vision = vision
        self.separation = separation

    def get_center(self, neighbors: Iterable):
        """Calculate center position of `neighbors`."""
        # NOTE: Center of neighbors in a toroidal space is quite ambiguous
        return vector_avg([neighbor.pos for neighbor in neighbors])

    def get_avg_heading(self, neighbors):
        """Calculate average direction of `neighbors`."""
        return vector_avg([neighbor.dir for neighbor in neighbors])

    def separator(self, neighbors: Iterable):
        """Generate vector that distances boids from each other."""
        # Init vector
        sep_dir = np.zeros(self.dim)
        # Loop over neighbors to see which are too close
        for neighbor in neighbors:
            distance_vector = self.model.space.get_distance_vector(self.pos, neighbor.pos)
            distance = np.linalg.norm(distance_vector)
            if distance < self.separation:
                # Seperator increases when boids get closer
                sep_dir += distance_vector * (self.separation / distance)
        return sep_dir

    def avoid_collision(self, obstacles):
        avoid_dir = np.zeros(self.dim)
        for obstacle in obstacles:
            headed_for_collision = obstacle.hit(*(self.pos + self.vision * normalize(self.dir)))
            collided = obstacle.hit(*self.pos)
            if collided or headed_for_collision:
                avoid_dir += self.model.space.get_distance_vector(self.pos, obstacle.pos)

        avoid_dir = normalize(avoid_dir)
        return avoid_dir

    def step(self) -> None:
        """Progress boid simulation by one step."""
        # Find boids that are whithin vision range.
        neighbors = filter_type(self.model.space.get_neighbors(self.pos, self.vision, False, torus=False), Boid)
        flock_dir = np.zeros(self.dim)
        separator_dir = np.zeros(self.dim)
        alignment_dir = np.zeros(self.dim)
        if len(neighbors) > 0:
            # Progress towards neighbors
            flock_dir += self.get_center(neighbors) - self.pos
            # Avoid collision with neighbors
            separator_dir += self.separator(neighbors)
            # Align direction with neighbors
            alignment_dir += self.get_avg_heading(neighbors)

        obstacles = filter_type(self.model.schedule.agents, Boid, include=False)
        avoid_dir = self.avoid_collision(obstacles)

        # Calculate change of direction, taking above parameters in account
        net_direction = flock_dir + separator_dir + alignment_dir
        # Avoiding objects is top priority
        if np.linalg.norm(avoid_dir) != 0:
            net_direction = avoid_dir
        # Limit turning speed by normalizing delta vector
        v_t = 0.1  # Turning speed
        delta_dir = v_t * normalize(net_direction)
        self.dir += delta_dir
        # Limit progression speed
        self.dir = normalize(self.dir)

        # Progress boid in model space
        new_pos = self.pos + self.dir * self.model.speed
        self.model.space.move_agent(self, new_pos)


class BoidModel(Model):
    """Class for boids simulation."""

    def __init__(
        self,
        N: int,
        box_size: np.ndarray,
        speed: float,
        separation: float = 1,
        do_collect: bool = True,
        obstacles: Iterable = None
    ):
        """Create new simulation."""

        # Set simulation parameters
        self.num_boids = N
        self.box_size = box_size
        self.dim = len(box_size)
        self.speed = speed
        self.separation = separation
        # Define iteration method
        self.schedule = SimultaneousActivation(self)
        # Define position space
        SpaceClass = ContinuousSpacePlus if self.dim != 3 else ContinuousSpace3D
        self.space = SpaceClass(*box_size, torus=True)
        # Initialize simulation
        self.do_collect = do_collect
        self.obstacles = obstacles
        self.make_agents()
        # Start animation
        self.running = True

    def make_agents(self):
        """Create agents."""

        # Create boids
        for i in range(self.num_boids):
            # Generate random position and direction
            pos = random_vector(self.dim) * self.box_size
            # Center direction vector around zero and normalize
            dir = random_vector(self.dim) - [0.5 for _ in range(self.dim)]
            dir = normalize(dir)

            # Initiate boid
            boid = Boid(i, self, pos, dir, separation=self.separation)
            self.space.place_agent(boid, pos)
            self.schedule.add(boid)

        # Create obstacles
        if self.obstacles:
            for i, _obstacle_parameters in enumerate(self.obstacles):
                obstacle_parameters = copy(_obstacle_parameters)
                # Start `unique_id`s at highest `unique_id` of boids
                id = i + self.num_boids
                obstacle_type = obstacle_parameters.pop('type')
                pos = obstacle_parameters['center']
                if obstacle_type == 'Rectangle':
                    obstacle = Rectangle(id, self, **obstacle_parameters)
                else:
                    continue
                self.space.place_agent(obstacle, pos)
                self.schedule.add(obstacle)

        if self.do_collect:
            # Activate data collection
            self.datacollector = DataCollector(
                agent_reporters={
                    "Position": "pos",
                    "Direction": "dir",
                },
                model_reporters={
                    "Flock count": count_flocks
                }
            )

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()


class Animation:
    """Animation class for displaying simulation to the user using matplotlib."""

    def __init__(self, fps=30, realtime: bool = False, **kwargs) -> None:
        self.do_realtime = realtime
        # Render realtime
        if self.do_realtime:
            assert "model" in kwargs, "Model must be passed for realtime animation."
            self.model = kwargs.get('model')
            i_max = kwargs.get('i_max')
            anim_func = self.realtime
            self.dim = len(self.model.box_size)
            self.box_size = self.model.box_size
        # Display animation with prerendered data
        else:
            assert "data" in kwargs, "Data must be passed for animation."
            self.agent_data, self.model_data = kwargs.get('data')
            i_max = kwargs.get('i_max', len(set(self.agent_data.index.get_level_values('Step'))))
            anim_func = self.prerendered
            self.dim = len(self.agent_data['Position'][0][0])
            self.box_size = kwargs.get('box_size')
            # Boid data are rows where direction is not None, since obstacles don't have a direction
            self.boid_data = self.agent_data.loc[self.agent_data['Direction'].notnull()]
            # With prerendered data, additional configuration information is required to draw obstacles
            self.render_config = kwargs['config']

        # Distinguish between 2D and 3D animation
        if self.dim == 2:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        elif self.dim == 3:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
        # Initiate animation
        self.animation = animation.FuncAnimation(self.fig,
                                                 anim_func,
                                                 init_func=self.init_anim,
                                                 frames=i_max,
                                                 interval=1000 / fps,
                                                 blit=True)

    def init_anim(self):
        """Set animation properties."""
        self.fig.suptitle("Boids simulation")

        # Distinguish between 2D and 3D animation
        if self.dim == 2:
            self.pos_plot, = self.ax.plot([], [], 'o', color='blue', animated=True)
            xlim, ylim = self.box_size
        elif self.dim == 3:
            self.ax = self.fig.gca(projection='3d')
            arr = np.empty(0)
            self.pos_plot, = self.ax.plot(arr, arr, arr, linestyle="",
                                          marker="o", color='blue',
                                          markersize=1.5,
                                          animated=True)
            xlim, ylim, zlim = self.box_size
            self.ax.set_zlim(0, zlim)
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)

        # Define obstacles from model
        self.rectangles = []
        if self.do_realtime:
            self.rectangles = filter_type(self.model.schedule.agents, Rectangle)
        # Define obstacles from config parameters
        else:
            rectangle_configs = self.render_config['obstacles']
            if rectangle_configs:
                for c in rectangle_configs:
                    if c['type'] == 'Rectangle':
                        DummyRectangle = namedtuple("ObjectName", c.keys())(*c.values())
                        self.rectangles.append(DummyRectangle)
        # Draw obstacles
        if len(self.rectangles) > 0:
            self.draw_rectangles(self.rectangles)

        plt.title(self._get_title(0))

        # Return figure object
        return self.pos_plot,

    def _get_title(self, flocks: int, i: int = None):
        """Generate animation title."""
        optional_s = "" if flocks == 1 else "s"
        title = f"{flocks} flock{optional_s}"
        if i:
            title = f"i: {i}\n" + title
        return title

    def realtime(self, i: int):
        """Update animation while running simulation concurrently."""
        # Draw initial frame at 0 steps
        if i != 0:
            self.model.step()

        # Retrieve relevant data from model
        boids = filter_type(self.model.schedule.agents, Boid)
        positions = [[boid.pos[i] for boid in boids] for i in range(self.dim)]
        directions = [[boid.dir[i] for boid in boids] for i in range(self.dim)]

        # Set title
        flocks = count_flocks(self.model)
        title = self._get_title(flocks, i)

        # rectangles = filter_type(self.model.schedule.agents, Rectangle)

        return self.update_plot(i, positions, directions, title)

    def prerendered(self, i: int):
        """Update animation with prerendered simulation data."""
        # Retrieve relevant data from data
        positions = [[boid[x] for boid in self.boid_data['Position'][i]] for x in range(self.dim)]
        directions = [[boid[x] for boid in self.boid_data['Direction'][i]] for x in range(self.dim)]

        # Set title
        flocks = self.model_data['Flock count'][i]
        title = self._get_title(flocks, i)

        return self.update_plot(i, positions, directions, title)

    def draw_rectangles(self, rectangles):
        """Draw rectangular obstacles in plot."""
        rect_artists = list()
        for rectangle in rectangles:
            width = rectangle.width
            height = rectangle.height
            rotation = rectangle.rotation
            bottom_left = (rectangle.center[0] - width / 2,
                           rectangle.center[0] - height / 2)
            rect = patches.Rectangle(bottom_left, width, height, rotation)
            rect_artists.append(self.ax.add_patch(rect))
        return rect_artists

    def update_plot(
        self, i: int,
        positions: Iterable,
        directions: Iterable,
        title: str,
        rectangles: Iterable = [],
        moving_obstacles: bool = False
    ):
        """Update animation plot with new data."""
        if i != 0:
            self.quivers.remove()

        plt.title(title)

        self.quivers = self.ax.quiver(*positions, *directions, animated=True)
        if self.dim == 2:
            self.pos_plot.set_data(*positions)
        elif self.dim == 3:
            self.pos_plot.set_data(*positions[0:2])
            self.pos_plot.set_3d_properties(positions[2])

        # Optionally draw moving obstacles
        rect_artists = list()
        if moving_obstacles:
            rect_artists = self.draw_rectangles(rectangles)

        # Return matplotlib artists
        return self.pos_plot, self.quivers, *rect_artists

    def show(self):
        """Display animation on screen."""
        plt.show()
        pass

    def save(self, output: str):
        """Save animation to `output`."""
        self.animation.save(output)


def read_json(file_path: str):
    """Read dictionary from json file at `file_path`."""
    with open(file_path, "r") as f:
        return json.load(f)
