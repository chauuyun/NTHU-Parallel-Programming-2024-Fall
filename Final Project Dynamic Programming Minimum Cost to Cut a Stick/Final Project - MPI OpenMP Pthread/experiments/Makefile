CC = gcc
CXX = g++
LDLIBS = -lpng -lm
CFLAGS = -O3 -g3
CXXFLAGS = $(CFLAGS)

final_pthread_v1: CFLAGS += -pthread -mavx512f -march=native
final_hybrid_v1 final_hybrid_v3 final_hybrid_v4: CC = mpicc
final_hybrid_v1 final_hybrid_v3 final_hybrid_v4: CXX = mpicxx
final_hybrid_v1 final_hybrid_v3 final_hybrid_v4: CFLAGS += -fopenmp -mavx512f -march=native
# 不再手動加入 -lmpi，mpicxx會自動鏈MPI庫

TARGETS = final_hybrid_v1 final_hybrid_v3 final_hybrid_v4 final_pthread_v1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
# module load icc openmpi
# ./judge.sh final_b t1000  