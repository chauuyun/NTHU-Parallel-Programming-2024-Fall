NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a
CXXFLAGS = -O3

LDFLAGS = -lm
EXES = final_baseline final_coalesce final_parallel_min_reduce final_first_reduce final_unroll

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

%: %.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

# module load cuda
# hw3-cat testcases/c01.1