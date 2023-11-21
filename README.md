# CUDA/ OpenMp Hybrid Big Bang N-Body Simulation

![bigbang](https://github.com/nrmancuso/big-bang/blob/main/gif/big-bang.gif)

### ABOUT

This program is adapted from Professor Stephen Siegel's sequential 2-d 
nbody simulation program, by Jiaman Wang and Nick Mancuso for our CISC372
final project. We have used CUDA and OpenMp to parallelize the original 
program and improve performance, so that we can create animations of
much larger galaxies.

The translation unit(s) that we have created are much larger than the original
ones, with the largest consisting of 72,000 bodies. Our animations were inspired
by the big bang, and the bodies that we have used simulate the birth of the 
milky way and andromeda galaxies. As the animation progresses, you can see 
the two galaxies forming.

You can find our animation here: https://vimeo.com/491779612

Much of our data was obtained from the research 
found at https://arxiv.org/abs/astro-ph/950901 by John Dubinsky, and liberally
manipulated the data for the best appearance in 2 dimensions. 

We have used the discretized universal gravitation equation provided in the 
sequential version of the program, with a few small modifications in the case
of the GPU friendly function.  Our algorithms/ logic stay fairly close to the
original sequential version, other than the parallelized sections. Each OpenMp 
thread has one of two responsiblities, depending on how the user chooses to
distribute the bodies (via `GPU_BODY_PROP`): 

(1) is the "manager" of one GPU device, or 
(2) is responsible for a block-distributed amount of bodies.

If an OpenMp thread is a "manager" of a GPU, this means that this thread will
call the CUDA kernel and handle most of the data copying responsibilities. 
If an OpenMP thread is resposible for updating the states of bodies, then 
it functions similarly to the sequential version for that thread's number of
owned bodies. 

When updating the states of bodies, each thread in the GPU is responsible 
for calculating the interactions between "it's" body and that of all other 
bodies; so essentially, one GPU thread is in charge of one body.

### PERFORMANCE

We have run numerous tests to determine that having the GPU handle ALL bodies
consistently provides the best performance. Please take a look at 
`nbody/graphs/big_bang_p100_diff-body-prop.pdf` to see the proportion of bodies updated
by the GPU increased vs. running time.  In this case, the OpenMP threads act
as managers for each GPU, handling memory copying and kernel calls.

Using one of the original translation units, `galaxy.c`, we have ran both strong 
and weak scaling experiments.  The results of these runs can be found in the 
following graphs:

`nbody/graphs/Galaxy_Strong_50-50-Efficiency.pdf`
`nbody/graphs/Galaxy_Strong_50-50-Speedup.pdf`
`nbody/graphs/Galaxy_Strong_50-50-Time.pdf`

`nbody/graphs/galaxy-weak-scaling-no-gpu-Efficiency.pdf`
`nbody/graphs/galaxy-weak-scaling-no-gpu-Speedup.pdf`
`nbody/graphs/galaxy-weak-scaling-no-gpu-Time.pdf`

All of the strong scaling used OpenMP threads (1 -> 16) and the body updates
were split 50/50 between the GPU and CPU, with 1001 bodies.

The weak scaling experiments increased the number of bodies by roughly 62 each time,
and all other variables were consistent with the strong scaling version.

In order to test which type of GPU was the most effective for this animation, we
tested 72,000 bodies on both types of GPU nodes, and threw in an MPI only run for 
good measure. Please see the `nbody/graphs/big_bang_p100vsk80vsMPI-time.pdf` file 
for more information.

Finally, we pitted two runs with different proportions of bodies being updated
by the CPU and GPU against each other in a weak scaling experiment.  The results
can be found in `nbody/graphs/big_bang_weak-scaling-50-90-1k80.pdf`.

### GETTING STARTED

The source code for our program is located in the `nbody/` directory.
There you will find all the translation units and the main driver program.
Additionally, there is a "configuration" file, config.c, where users can specify
additional/ different colors to produce for the animations.

To test our version of nbody, `nbody_omp.cu`, against the original sequential
version of nbody:

```````````````````````````
$ make test

```````````````````````````

This will use the original translation units to produce both parallel and 
sequential versions of each animation, then `diff` the results. Note that
depending on which machine you are testing on and how many GPU's you want
 to use, you will want to comment/ uncomment the makefile accordingly:

```````````````````````````

	# below is for single gpu on grendel
	OMP_NUM_THREADS=$(NPROCS) $(CUDARUN) ./$< $@ $(GPU_BODY_PROP)
	# below is for two gpus on grendel
	#OMP_NUM_THREADS=$(NPROCS) srun --unbuffered -n 1 --gres=gpu:2 ./$< $@ $(GPU_BODY_PROP)
	# below is for bridges interactive mode
	#./$< $@ $(GPU_BODY_PROP)

`````````````````````````````

The GPU_BODY_PROP is the proportion of bodies that the GPU will update.  This
proportion can be adjusted at the top of the makefile.

In order to complie and link all of our translation units, do:

````````````````````````````
make big_bangs

````````````````````````````

When running compiled/ linked translation units manually, you must specify 
the proportion of bodies that the GPU will update via command line arg:

`````````````````````````````
./big_bang8.exec big_bang8.anim 0.50

`````````````````````````````
This would run big_bang8.exec, create an animation called big_bang8.anim, and
the GPU would update half of the bodies. To see more usage examples of our 
program, you can check out the bridges scripts, found in nbody/graphs/bridges.  The
range of GPU_BODY_PROP is from 0.0 to 1.0, inclusive. 

In general, the syntax for the executables generated from the translation units
is:

`````````````````````````````
<machine-specific env variables> ./<translation unit name>.exec <output file name>.anim <gpu bodies proportion>

`````````````````````````````

NOTE: This program MUST be linked to translation units; it cannot be ran
on it's own!
