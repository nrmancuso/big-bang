# Makefile for nbody graviational simulation, MPI version
# edit ROOT as needed to point to 372-2020F/code
ROOT = ../../372-2020F/code
SNAME = nbody
NAME = $(SNAME)_omp
SDIR = $(SEQ)/$(SNAME)
NPROCS = 16
#Below is the proportion of bodies that the GPU will update
GPU_BODY_PROP = 1.0
include $(ROOT)/common.mk
# Larger runs (more nbodies) are commented
#     out so that we can test incrementally
bigbangs = big_bang1 big_bang2 big_bang3 big_bang4 big_bang
stems1 = planets-elliptical figure8 galaxy #galaxy2
stems =  $(stems1)
movies1 = $(addsuffix .mp4,$(stems1))
movies = $(addsuffix .mp4,$(stems))
execs1 = $(addsuffix .exec,$(stems1))
execs = $(addsuffix .exec,$(stems))
anims1 = $(addsuffix .anim,$(stems1))
anims = $(addsuffix .anim,$(stems))
tests1 = $(addsuffix .test,$(stems1))
tests = $(addsuffix .test,$(stems))

all: $(execs)

test:
	# here we only test the original translation 
	#    units, to ensure versions are identical
	$(MAKE) -C $(SDIR) $(anims1)
	$(MAKE) $(tests1)

big_bangs:
	$(NVCCC) -O3 -c --fmad true -I$(SDIR)  -o $(NAME).o $(NAME).cu \
	-Xcompiler -fopenmp
	$(NVCCC) -O3 -I$(SDIR) -I../../372-2020F/code/src/seq/nbody -o big_bang1.exec \
		big_bang1.c nbody_omp.o -lanim -lm -lgomp
	$(NVCCC) -O3 -I$(SDIR) -I../../372-2020F/code/src/seq/nbody -o big_bang2.exec \
		big_bang2.c nbody_omp.o -lanim -lm -lgomp
	$(NVCCC) -O3 -I$(SDIR) -I../../372-2020F/code/src/seq/nbody -o big_bang3.exec \
		big_bang3.c nbody_omp.o -lanim -lm -lgomp
	$(NVCCC) -O3 -I$(SDIR) -I../../372-2020F/code/src/seq/nbody -o big_bang4.exec \
		big_bang4.c nbody_omp.o -lanim -lm -lgomp
	$(NVCCC) -O3 -I$(SDIR) -I../../372-2020F/code/src/seq/nbody -o big_bang5.exec \
		big_bang5.c nbody_omp.o -lanim -lm -lgomp
	$(NVCCC) -O3 -I$(SDIR) -I../../372-2020F/code/src/seq/nbody -o big_bang6.exec \
		big_bang6.c nbody_omp.o -lanim -lm -lgomp
	$(NVCCC) -O3 -I$(SDIR) -I../../372-2020F/code/src/seq/nbody -o big_bang7.exec \
		big_bang7.c nbody_omp.o -lanim -lm -lgomp
	$(NVCCC) -O3 -I$(SDIR) -I../../372-2020F/code/src/seq/nbody -o big_bang8.exec \
		big_bang8.c nbody_omp.o -lanim -lm -lgomp
	
$(bigbangs): Makefile
	$(MAKE) $@.exec
	#$(CUDARUN) ./$@.exec $@.anim
	# We really don't always need to make all animations for each time we 
	# test, so we will leave the next line commented out unless we need it.
	#$(RUN) anim2mp4 $@.anim

$(tests): %.test: %.anim
	$(RUN) diff $(SDIR)/$< $<

$(NAME).o: Makefile $(SDIR)/$(SNAME).h $(NAME).cu $(ANIM)
	$(NVCCC) -c --fmad false -I$(SDIR)  -o $(NAME).o $(NAME).cu \
	-Xcompiler -fopenmp

$(execs): %.exec: $(SDIR)/%.c Makefile $(NAME).o $(ANIM) $(SDIR)/$(SNAME).h
	$(NVCCC) -O3 -I$(SDIR) -o $@ $< $(NAME).o -lanim -lm -lgomp

$(anims): %.anim: %.exec
	# below in for single gpu on grendel
	OMP_NUM_THREADS=$(NPROCS) $(CUDARUN) ./$< $@ $(GPU_BODY_PROP)
	# below is for two gpus on grendel
	#OMP_NUM_THREADS=$(NPROCS) srun --unbuffered -n 1 --gres=gpu:2 ./$< $@ $(GPU_BODY_PROP)
	# below is for bridges interactive mode
	#./$< $@ $(GPU_BODY_PROP)

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test $(tests) $(bigbangs)
