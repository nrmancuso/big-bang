#include "nbody.h"
#define RADIUS 2500000
#define mway_galaxy_disc 3
#define and_galaxy_disc 3
#define mway_galaxy_bulge 2
#define and_galaxy_bulge 2
#define mway_galaxy_halo 1
#define and_galaxy_halo 1

const double x_min = -RADIUS, x_max = RADIUS, y_min = -RADIUS, y_max = RADIUS;
const int nx = 1000;
// number of frames is 10000/5 = 2000.  At 60 fps, that's a
// 2000/60=33.3333s movie.
const int nstep = 10000;
const int wstep = 5;
const int ncolors = 6;
const double delta_t = 250;

const int colors[][3] = {
        {255, 255, 255},  // white
        {245, 245, 245},  // very light grey
        {153, 204, 255},  // light blue
        {0, 0, 139},      //
        {51, 51, 153},    // dark blue
        {51, 51, 153},    // dark blue
};
