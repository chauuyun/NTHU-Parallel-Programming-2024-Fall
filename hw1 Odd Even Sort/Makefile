CC = mpicc
CXX = mpicxx
CXXFLAGS = -O3 -lm
CFLAGS = -O3 -lm
export OMPI_CXX = icc
CXXFLAGS += -fp-model precise 

hw1NVTX: CXXFLAGS += -DNVTX
hw1NVTX: hw1
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

# module load openmpi compiler icc

