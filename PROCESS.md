# Process notes

Chronological goals:
- [x] Implement objects with velocity vectors and iterate movement
    - [x] Spawn boids with random position and random direction
    - [x] Visualize boids with velocity vectors
    - [x] Implement movement according to headings
    - [x] Animate boids movement
- [x] Implement collision detection and avoidance/seperation of fellow boids
- [x] Implement alignment of boids
- [x] Implement cohesion (flocking)
- [x] 3D expansion
    - [x] Rewrote mesa.space.ContinuousSpace module for 3D expansion
    - [x] Matplotlib 3D expansion
- [x] Implement caching of output data to pickles by recognizing a configuration with which a simulation as been run previously. 
    - [x] Implement dictionary/configuration hashing to string
- [x] Define what a 'flock' is and keep track  of the amount of flocks over time with variation of parameters
    - [x] Decide between heuristical approach and exact: heuristic works good enough
- [x] Write code to repeat simulation with same parameters for data mining
- [x] Write code to repeat simulation with varied parameters for comparison
- [x] Fix flocks turning around due to toroidal space
    - explanation: due to boids looking beyond the box's borders for neighbors, they find neighbors that are on the other side of the box, thus conceiving the center of the flock to be nearer to the center of the box rather than their actual vicinity. Disabling toroidial space in space.get_neighbors() fixes this, but then both ContinuousSpace3D and ContinuousSpace need a rewrite.
- [x] Implement object collision avoidance (2D)