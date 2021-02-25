#setup
set terminal pdf

# 2 p100's vs 4 k80's, scaling number of bodies
set output "graphs/big_bang_p100vsk80vsMPI-time.pdf"
set title "Scaling number of bodies, 2 p100's vs 4 k80's vs pure MPI"
set xlabel center "number of bodies"
set ylabel center "time (seconds)"
set xr [0:74000]
set yr [0:4000]
plot "data_files/big_bang_p100.dat" using 1:2 title "2 p100 GPUs" with lines,  \
    "data_files/big_bang_k80.dat" using 1:2 title "4 k80 GPU's" with lines, \
    "data_files/mpi.out.dat" using 1:2 title "MPI only" with lines


# 2 p100's, different proportions of bodies updated on the cpu/gpu
set output "graphs/big_bang_p100_diff-body-prop.pdf"
set title "Proportions of bodies updated by GPU vs running time"
set xlabel center "proportion of bodies updated by GPUs"
set ylabel center "time(seconds)"
set xr[0.5:1]
set yr[0:8000]

plot "data_files/big_bang_inc_nbodies.dat" using 1:2 title "Efficiency of increasing proportion of bodies\n updated on GPUs" with lines

# weak scaling big_bang at 50% vs weak scaling at 90%, 1 k80 GPU
set output "graphs/big_bang_weak-scaling-50-90-1k80.pdf"
set title "Weak Scaling big bang, 1 k80 GPU"
set ylabel center "time(seconds)"
set xlabel center "number of OMP threads"
set xr[0:16]
set yr[0:200]

plot "data_files/big_bang5_weak_90prcntGPU.out.dat" using 1:3 title "90% of bodies updated on GPU" with lines, \
    "data_files/big_bang5_weak_50prcntGPU.out.dat" using 1:3 title  "50% of bodies updated on GPU" with lines

###### galaxy weak scaling NO GPU ######
# speedup
set output "graphs/galaxy-weak-scaling-Speedup.pdf"
set title "Galaxy Weak Scaling Speedup"
set xlabel center "Number of processes"
set ylabel center "speedup"
set xr [0:16]
set yr [0:0.10]
#first(x) = ($0 > 0 ? base : base = x)
plot "data_files/galaxy_weak_fullgpuANDseq.dat" using 1:($4, $4/$2) title "Speedup" with linespoints
# efficiency
set output "graphs/galaxy-weak-scaling-Efficiency.pdf"
set title "Galaxy Weak Scaling Efficiency"
set xlabel center "Number of processes"
set ylabel center "efficiency"
set xr [0:20]
set yr [0:0.02]
#first(x) = ($0 > 0 ? base : base = x)
plot "data_files/galaxy_weak_fullgpuANDseq.dat" using 1:($4, $4/($2*$1)) title "Efficiency" with linespoints
# time
set output "graphs/galaxy-weak-scaling-Time.pdf"
set title "Galaxy Weak Scaling Time"
set xlabel center "Number of bodies"
set ylabel center "time (seconds)"
set xr [0:1000]
set yr [0:50]
plot "data_files/galaxy_weak_fullgpuANDseq.dat" using 2:3 title "CUDA time" with linespoints, \
    "data_files/galaxy_weak_fullgpuANDseq.dat" using 2:4 title "Seq time" with linespoints

#### galaxy strong scaling #######
# galaxy strong scaling speedup
set output "graphs/Galaxy_Strong_50-50-Speedup.pdf"
set title "Galaxy Strong Scaling Speedup"
set xlabel center "Number of processes"
set ylabel center "speedup"
set xr [0:20]
set yr [0:2]
first(x) = ($0 > 0 ? base : base = x)
plot "data_files/galaxy_strong.out.dat" using 1:(first($2), base/$2) title "Speedup" with linespoints
# galaxy strong scaling efficiency
set output "graphs/Galaxy_Strong_50-50-Efficiency.pdf"
set title "Galaxy Strong Scaling Efficiency"
set xlabel center "Number of processes"
set ylabel center "efficiency"
set xr [0:20]
set yr [0:1]
first(x) = ($0 > 0 ? base : base = x)
plot "data_files/galaxy_strong.out.dat" using 1:(first($2), base/($2*$1)) title "Efficiency" with linespoints
# galaxy strong scaling time
set output "graphs/Galaxy_Strong_50-50-Time.pdf"
set title "Galaxy Strong Scaling Time"
set xlabel center "Number of processes"
set ylabel center "time (seconds)"
set xr [0:20]
set yr [0:100]
plot "data_files/galaxy_strong.out.dat" using 1:2 title "Time" with linespoints

#### k80 scaling number of GPUs #####
# speedup
set output "graphs/k80-big_bang_incGPUs-Speedup.pdf"
set title "big bang k80 GPU scaling Speedup"
set xlabel center "Number of GPU's"
set ylabel center "speedup"
set xr [0:5]
set yr [0:6]
first(x) = ($0 > 0 ? base : base = x)
plot "data_files/k80_GPU_scaling.dat" using 1:(first($2), base/$2) title "Speedup" with linespoints

# time
set output "graphs/k80-big_bang_incGPUs-Time.pdf"
set title "big bang k80 GPU scaling Time"
set xlabel center "Number of GPU's"
set ylabel center "time (seconds)"
set xr [0:5]
set yr [0:5000]
plot "data_files/k80_GPU_scaling.dat" using 1:2 title "Time" with linespoints
