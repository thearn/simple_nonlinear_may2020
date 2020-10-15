
Simple non-linear airspace trajectories with Dymos
===================================================

This is a 2D multi-agent single-integrator EOM model, with vectorized positional states `X` and `Y`, and with controls `Vx` and `Vy`. The purpose of this model is to assess and benchmark strategies for handling separation constraints with large numbers of vehicles/agents in a dynamic airspace model.

With the right parameterization, this model generates a 2D airspace, with each agent having a designated starting and ending location. The problem then minimizes the time needed for all agents to reach their destination, while respecting separation contraints between each other.

`airspace_alldist.py` is the main run file. This runs the model, creates a plot of the results, and outputs a data file `flights.dat`. You can then run `movie.py` to create an mp4 movie that animates the result (ffmpeg is required).

Parameters
-------------

At the bottom, you'll find:

```
if __name__ == '__main__':
    generate_airspace(nv=8, # number of vehicles
                      ns=25, # number of sample points for dymos
                      limit=100.0, # separation limit (in km)
                      airspace_type = 0, # 0 = square region, low interaction. 1 = circular region, high interaction
                      separation='grid', # separation method. 'grid', 'pairwise', or 'none'
                      aggregate='mine', # separation constraint aggregation. 'mine', 'ks', or 'none'
                      seed=1)# random seed for numpy
```

`nv` is the number of vehicles/agents.

`ns` is the number of sample points for Dymos.

`limit` is the separation limit for Dymos, in km. i.e. a constraint will be included to keep vehicles apart from each other at least this amount, at all times.

`airspace_type` specifies the spatial layout of the problem (i.e. how starting/ending points are arranged spatially). `airspace_type = 0` is a rectangular layout with origin -> destination location placed randomly. This leads to a fairly low amount of agent interaction.
`airspace_type = 1` is a circular layout, with origin -> destination locations placed around the edge of the circle. This tends to lead to a large amount of agent interaction, and is more difficult to converge. But is a good upper-limit benchmark for the cost of computing separation constraints.

`separation` is the method for computing separation constraints. `separation='none'` runs the model without any separation constraints (to baseline or debug the EOM). `separation='pairwise'` computes the separation between each pair of vehicles/agents (the direct approach). `separation='grid'` computes the separation using my method.

`aggregate` specifies the kind of aggregation to use for the separation constraints. 
This really only comes into play when `separation='pairwise'`. Choices are 'none' or 'ks'.
When using `separation='grid'` (my method), the only valid option is the built-in aggregation, 
`aggregate = 'mine'`.

Example results:
-----------------
8 agent model in rectangular airspace converges in 6.645s:

```
nv=8
ns=25
limit=100.0 
airspace_type=0 
separation='grid'
aggregate='mine'
seed=1
```

![Example 1](results/example1.png)

--------------

Same model, but with pairwise separation and KS aggregation. Takes 9.6s:

```
nv=8
ns=25
limit=100.0 
airspace_type=0 
separation='pairwise'
aggregate='ks'
seed=1
```

![Example 1](results/example1_ks.png)

--------------

--------------
Same model again, but with pairwise separation and no aggregation. Takes 17.3s:

```
nv=8
ns=25
limit=100.0 
airspace_type=0 
separation='pairwise'
aggregate='none'
seed=1
```

![Example 1](results/example1_pw.png)



--------------

15 agent model in circular airspace, converges in 45.64s:

```
nv=15
ns=25
limit=100.0 
airspace_type=1 
separation='grid'
aggregate='mine'
seed=1
```

![Example 1](results/example2.png)