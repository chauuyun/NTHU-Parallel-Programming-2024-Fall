NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a
CXXFLAGS = -O3

LDFLAGS = -lm
EXES = final final_cpu final_seq

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

final_cpu: final_cpu.cpp
	g++ $(CXXFLAGS) -o $@ $?

final_seq: final_seq.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

final: final.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

# module load cuda
# hw3-cat testcases/c01.1