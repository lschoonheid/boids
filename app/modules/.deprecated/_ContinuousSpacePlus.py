"""
Copy of Mesa ContinuousSpace class with 3D expansion and options for 
toroidal space in spatial methods.
"""

from mesa import Agent
from mesa.space import ContinuousSpace, GridContent
import numpy as np

from typing import (
    List,
    Union,
    Tuple
)

from modules.helpers import min_norm


FloatCoordinate = Union[Tuple[float, float, float], np.ndarray]


class ContinuousSpacePlus(ContinuousSpace):
    def _do_torus(self, torus_self, torus_arg):
        return ((torus_self and torus_arg is None) or torus_arg) and not torus_arg is False

    def get_neighbors(
        self, pos: FloatCoordinate, radius: float, include_center: bool = True, torus = None
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

    def get_distance(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate, torus = None) -> float:
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
    
    def get_distance_vector(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate, torus = None) -> float:
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
    """Continuous space where each agent can have an arbitrary position.

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
                      include_center: bool = True, torus = None
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

    def get_distance(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate, torus = None) -> float:
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

    def get_distance_vector(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate, torus = None) -> float:
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
