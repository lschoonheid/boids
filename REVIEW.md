Things to improve: 
* Separate helper module into separate files for cleanliness
* Obstacles
    * Take what I did for `Rectangle(Obstacle)` and expand it to `Cube(Rectangle)` so it works in three dimensions. All I would have to do is add the `z` variable at each spatial operation.
    * Add `Ovals` with roughly the same method of `Rectangle(obstalce)`.
* Make quivers/vectors draw in 3D-Matplotlib, although this may simply be a limitation of the not so expansive 3D-matplotlib capabilities.
* Tidier data separation of boid data, model data and obstacle data. It is now all thrown on one heap, which is handy to retrieve the total simulation information, but it could be made easier to filter.
* Add an option to save the data when running the simulation in realtime
* Also implement multiprocessing for `compare.py`
* Make simulation faster by implementing a [`QuadTree`](https://en.wikipedia.org/wiki/Quadtree) structure or [`spatial hashing`](https://www.gamedev.net/tutorials/programming/general-and-gameplay-programming/spatial-hashing-r2697/). This way you don't have neighboring checks in `O(n^2)`.