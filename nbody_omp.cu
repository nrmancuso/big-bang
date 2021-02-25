/* nbody.c: sequential 2-d nbody simulation
   Author: Nick Mancuso and Jiaman Wang

   Link this with a translation unit that defines the extern
   variables, and anim.o, to make a complete program.
 */
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
//#include "anim_dev.h"
//#include "anim.h"
#include "cudaanim.h"
#include "nbody.h"

/* Global variables */
extern const double x_min;             /* coord of left edge of window (meters) */
extern const double x_max;             /* coord of right edge of window (meters) */
extern const double y_min;             /* coord of bottom edge of window (meters) */
extern const double y_max;             /* coord of top edge of window (meters) */
extern const int nx;                   /* width of movie window (pixels) */
extern const int nbodies;              /* number of bodies */
extern const double delta_t;           /* time between discrete time steps (seconds) */
extern const int nstep;                /* number of time steps */
extern const int wstep;                /* number of times steps beween movie frames */
extern const int ncolors;              /* number of colors to use for the bodies */
extern const int colors[][3];          /* colors we will use for the bodies */
extern const Body bodies[];            /* list of bodies with initial data */
__device__ const double G = 6.674e-11; /* universal gravitational constant */
int ny;                                /* height of movie window (pixels) */
State* states;                         /* host copy of state array */
ANIM_File af;                          /* output anim file */
double* posbuf;                        /* to send data to anim, 2*nbodies doubles */
double start_time;                     /* time simulation starts */
int num_gpu_devices;                   /* number of gpus */
int cpu_nbodies;                       /* number of bodies that the cpu is updating */
int gpu_nbodies;                       /* number of bodies that the gpu is updating */

// Below, we declare all as thread private so that each thread maintains these
// pointers across all parallel regions for the life of the thread.
State* states_dev;
#pragma omp threadprivate(states_dev)
Body* bodies_dev;
#pragma omp threadprivate(bodies_dev)
int err;
#pragma omp threadprivate(err)
State* states_new_dev;
#pragma omp threadprivate(states_new_dev)
State* states_new;
#pragma omp threadprivate(states_new)
int num_owned;
#pragma omp threadprivate(num_owned)
int first;
#pragma omp threadprivate(first)
int cpu_thread_id;
#pragma omp threadprivate(cpu_thread_id)
int num_cpu_threads;
#pragma omp threadprivate(num_cpu_threads)
int nblocks;
#pragma omp threadprivate(nblocks)

// Set up distribution here, these are more complicated than usual
//  due to division of labor between GPU and CPU
#define threadsPerBlock 1024
#define FIRST(device, num_gpu_devices_or_threads, nbodies) \
    ((((ulong)(device)) * ((ulong)nbodies)) / num_gpu_devices_or_threads)
#define NUM_OWNED(device, num_gpu_devices_or_threads, nbodies) \
    (FIRST(device + 1, num_gpu_devices_or_threads, nbodies) -  \
     FIRST(device, num_gpu_devices_or_threads, nbodies))

/* Handles setup */
static void init(char* filename, float gpu_body_prop) {
    start_time = ANIM_time();
    assert(x_max > x_min && y_max > y_min);
    ny = ceil(nx * (y_max - y_min) / (x_max - x_min));
    printf("nbody: nbodies=%d nx=%d ny=%d nstep=%d wstep=%d\n", nbodies, nx, ny, nstep,
           wstep);
    const int nframes = wstep == 0 ? 0 : 1 + nstep / wstep;
    printf("nbody: creating ANIM file %s with %d frames, %zu bytes.\n", filename, nframes,
           ANIM_Nbody_file_size(2, nbodies, ncolors, nframes));
    fflush(stdout);
    assert(gpu_body_prop >= 0.0 && gpu_body_prop <= 1.0);
    assert(nx >= 10 && ny >= 10);
    assert(nstep >= 1 && wstep >= 0 && nbodies > 0);
    assert(ncolors >= 1 && ncolors <= ANIM_MAXCOLORS);
    states = (State*)malloc(nbodies * sizeof(State));
    assert(states);
    posbuf = (double*)malloc(2 * nbodies * sizeof(double));
    assert(posbuf);

    // Init device memory for state and body arrays
    cudaGetDeviceCount(&num_gpu_devices);

    // This is where we divide up the work between the cpu and gpu
    gpu_nbodies = nbodies * gpu_body_prop;
    cpu_nbodies = nbodies - gpu_nbodies;

    // Here we set the number of threads to be the same as the number of GPUs,
    // if the cpu isn't updating any bodies. This provides the best performance.
    if (cpu_nbodies == 0) {
        omp_set_num_threads(num_gpu_devices);
    }

    printf(
        "Running nbody with %d GPU devices. The gpu is updating %d nbodies and the cpu "
        "is updating %d nbodies (gpu_body_prop = %.2f).\n",
        num_gpu_devices, gpu_nbodies, cpu_nbodies, gpu_body_prop);

    // Set up animation
    assert(states);
    int radii[nbodies], bcolors[nbodies];
    ANIM_color_t acolors[ncolors];  // RGB colors converted to ANIM colors

    // Setup arrays to pass to ANIM_Create_nbody
#pragma omp parallel for shared(bodies, states, radii, bcolors)
    for (int i = 0; i < nbodies; i++) {
        assert(bodies[i].mass > 0);
        assert(bodies[i].color >= 0 && bodies[i].color < ncolors);
        assert(bodies[i].radius > 0);
        states[i] = bodies[i].state;
        radii[i] = bodies[i].radius;
        bcolors[i] = bodies[i].color;
    }

    // These declarations must be explicit in CUDA
    int dims[] = {nx, ny};
    ANIM_range_t ranges[] = {{x_min, x_max}, {y_min, y_max}};

    // Translate color codes to ANIM colors
#pragma omp parallel for shared(acolors, colors)
    for (int i = 0; i < ncolors; i++)
        acolors[i] = ANIM_Make_color(colors[i][0], colors[i][1], colors[i][2]);

    // Initialize anim file
    af = ANIM_Create_nbody(2, dims, ranges, nbodies, radii, ncolors, acolors, bcolors,
                           filename);
}

/* Writes a frame to the position buffer */
static inline void write_frame() {
#pragma omp parallel for shared(states, posbuf)
    for (int i = 0; i < nbodies; i++) {
        posbuf[2 * i] = states[i].x;
        posbuf[2 * i + 1] = states[i].y;
    }

    ANIM_Write_frame(af, posbuf);
}

/* Move forward one time step.  This is the "integration step".  For
   each body b, compute the total force acting on that body.  If you
   divide this by the mass of b, you get b's acceleration.  So you
   actually just calculate b's acceleration directly, since this is
   what you want to know.  Once you have the acceleration, update the
   velocity, then update the position.

   int first - the index of this GPU's first state to update
   int num_owned - the total number of states that this GPu is updating
   double delta_t - the change in time
   State* states_dev - the main states array, in total
   State* states_new_dev - this GPU's new state array (size of num_owned)
   Body* bodies - the intial Body array, in total
   int device_id - the gpu id of this gpu
   int nblocks - the number of blocks declared for this GPU
   int nbodies - the total number of nbodies

*/
__global__ static void update(int first, int num_owned, double delta_t, State* states_dev,
                              State* states_new_dev, Body* bodies, int device_id,
                              int nblocks, int nbodies) {
    // Get id of this CUDA thread
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_idx = local_idx + first;

    // Set up variables for calculation
    double x = states_dev[global_idx].x, y = states_dev[global_idx].y;
    double vx = states_dev[global_idx].vx, vy = states_dev[global_idx].vy;
    // ax times delta t, ay times delta t...
    double ax_delta_t = 0.0, ay_delta_t = 0.0;

    if (global_idx >= first && global_idx < first + num_owned) {
#ifdef DEBUG
        printf("global_idx (%d) from device (%d) in update!\n", global_idx, device_id);
        printf("x = %.2f, y = %.2f, vx = %.2f, vy = %.2f\n", x, y, vx, vy);
#endif
        for (int j = 0; j < nbodies; j++) {
            if (j == global_idx) continue;  // should be global
            const double dx = states_dev[j].x - x, dy = states_dev[j].y - y;
            const double mass = bodies[j].mass;

            const double r_squared = dx * dx + dy * dy;

            if (r_squared != 0) {
                const double r = sqrt(r_squared);

                if (r != 0) {
                    const double acceleration = G * mass / r_squared;
                    const double atOverr = acceleration * delta_t / r;

                    ax_delta_t += dx * atOverr;
                    ay_delta_t += dy * atOverr;
                }
            }
        }
        vx += ax_delta_t;
        vy += ay_delta_t;
        x += delta_t * vx;
        y += delta_t * vy;

        assert(!isnan(x) && !isnan(y) && !isnan(vx) && !isnan(vy));
        states_new_dev[local_idx] = (State){x, y, vx, vy};
    }
}

/* Move forward one time step.  This is the "integration step".  For
   each body b, compute the total force acting on that body.  If you
   divide this by the mass of b, you get b's acceleration.  So you
   actually just calculate b's acceleration directly, since this is
   what you want to know.  Once you have the acceleration, update the
   velocity, then update the position.

   int first - the index of this GPU's first state to update
   int my_nbodies - the number of bodies that this thread is updating

*/
static inline void cpu_update(int first, int my_nbodies) {
    int states_new_index = 0;
    for (int i = first; i < first + my_nbodies; i++) {
        double x = states[i].x, y = states[i].y;
        double vx = states[i].vx, vy = states[i].vy;
        // ax times delta t, ay times delta t...
        double ax_delta_t = 0.0, ay_delta_t = 0.0;

        for (int j = 0; j < nbodies; j++) {
            if (j == i) continue;

            const double dx = states[j].x - x, dy = states[j].y - y;
            const double mass = bodies[j].mass;
            const double r_squared = dx * dx + dy * dy;

            if (r_squared != 0) {
                const double r = sqrt(r_squared);

                if (r != 0) {
                    const double acceleration = G * mass / r_squared;
                    const double atOverr = acceleration * delta_t / r;

                    ax_delta_t += dx * atOverr;
                    ay_delta_t += dy * atOverr;
                }
            }
        }
        vx += ax_delta_t;
        vy += ay_delta_t;
        x += delta_t * vx;
        y += delta_t * vy;
        assert(!isnan(x) && !isnan(y) && !isnan(vx) && !isnan(vy));
        states_new[states_new_index++] = (State){x, y, vx, vy};
    }
}

/* Close GIF file, free all allocated data structures */
static void wrapup() {
    ANIM_Close(af);
    free(posbuf);
    free(states);
#pragma omp parallel
    {
        free(states_new);
        if (cpu_thread_id < num_gpu_devices) {
            int gpu_id = -1;
            cudaSetDevice(cpu_thread_id % num_gpu_devices);
            cudaGetDevice(&gpu_id);
            cudaFree(states_dev);
            cudaFree(bodies_dev);
            cudaFree(states_new_dev);
        }
    }

    printf("\nnbody: finished.  Time = %lf\n", ANIM_time() - start_time);
}

/* One argument: the name of the output file */
int main(int argc, char* argv[]) {
    int statbar = 0;  // used for printing status updates
    assert(argc == 3);
    float gpu_body_prop = atof(argv[2]);
    init(argv[1], gpu_body_prop);

// Initialize memory for all threads and GPUs
#pragma omp parallel shared(num_gpu_devices, gpu_nbodies, cpu_nbodies)
    {
        // Set up individual thread variables
        cpu_thread_id = omp_get_thread_num();
        num_cpu_threads = omp_get_num_threads();

        // If cpu is in charge of GPU updates
        if (cpu_thread_id < num_gpu_devices && num_gpu_devices > 0) {
            num_owned = NUM_OWNED(cpu_thread_id, num_gpu_devices, gpu_nbodies);
            first = FIRST(cpu_thread_id, num_gpu_devices, gpu_nbodies);
            // Assign and check the GPU device for each thread
            int gpu_id = -1;
            cudaSetDevice(cpu_thread_id);
            cudaGetDevice(&gpu_id);

            // Set number of blocks in grid for this GPU
            nblocks = num_owned / (threadsPerBlock * num_cpu_threads) +
                      (0 != (num_owned % (threadsPerBlock * num_cpu_threads)));
#ifdef DEBUG
            printf("gpu thread %d, num owned = %d, first = %d\n", cpu_thread_id,
                   num_owned, first);
#endif
            // Allocate memory for GPU
            if (num_owned > 0) {
                err = cudaMalloc((void**)&states_dev, nbodies * sizeof(State));
                assert(err == cudaSuccess);
                err = cudaMalloc((void**)&bodies_dev, nbodies * sizeof(Body));
                assert(err == cudaSuccess);
                err = cudaMalloc((void**)&states_new_dev, num_owned * sizeof(State));
                assert(err == cudaSuccess);

                // Only do this once, since this is never updated
                err = cudaMemcpy(bodies_dev, bodies, nbodies * sizeof(Body),
                                 cudaMemcpyHostToDevice);
                assert(err == cudaSuccess);

                // Host copies for new state array for each thread
                states_new = (State*)malloc(num_owned * sizeof(State));
            }
        }
        // If cpu does updates itself
        else if (cpu_nbodies > 0) {
            int offset_cpu_id = cpu_thread_id - num_gpu_devices;
            int offset_num_gpus = num_cpu_threads - num_gpu_devices;

            num_owned = NUM_OWNED(offset_cpu_id, offset_num_gpus, cpu_nbodies);

            // Below we are adding the offset from the bodies that the
            //     gpu is updating
            first = FIRST(offset_cpu_id, offset_num_gpus, cpu_nbodies) + gpu_nbodies;
#ifdef DEBUG
            printf("cpu thread %d, num owned = %d, first = %d\n", cpu_thread_id,
                   num_owned, first);
#endif

            if (num_owned > 0) states_new = (State*)malloc(num_owned * sizeof(State));
        }
    }

    if (wstep != 0) write_frame();
    for (int i = 1; i <= nstep; i++) {
// Call device kernel from each thread
#pragma omp parallel shared(num_gpu_devices, states)
        {
            // Call update for GPUs
            if (cpu_thread_id < num_gpu_devices && num_gpu_devices > 0) {
                // Assign and check the GPU device for each thread
                int gpu_id = -1;
                cudaSetDevice(cpu_thread_id);
                cudaGetDevice(&gpu_id);

                if (num_owned > 0) {
                    err = cudaMemcpy(states_dev, states, nbodies * sizeof(State),
                                     cudaMemcpyHostToDevice);
                    assert(err == cudaSuccess);

                    update<<<nblocks, threadsPerBlock>>>(
                        first, num_owned, delta_t, states_dev, states_new_dev, bodies_dev,
                        gpu_id, nblocks, nbodies);

                    // Copy back this gpu's part of states_new_dev back to host's states
                    // into correct location
                    err = cudaMemcpy(states_new, states_new_dev,
                                     num_owned * sizeof(State), cudaMemcpyDeviceToHost);
                    assert(err == cudaSuccess);
                }
            }
            // Here we call the cpu to update any bodies that it owns
            else if (cpu_nbodies > 0) {
                if (num_owned > 0) cpu_update(first, num_owned);
            }

#pragma omp barrier

            // If this thread owns nbodies, then update states
            if (num_owned > 0)
                memcpy(&states[first], states_new, num_owned * sizeof(State));

        }  // Implicit barrier for all threads

        ANIM_Status_update(stdout, nstep, i, &statbar);
        if (wstep != 0 && i % wstep == 0) {
            write_frame();
        }
    }
    wrapup();
}
