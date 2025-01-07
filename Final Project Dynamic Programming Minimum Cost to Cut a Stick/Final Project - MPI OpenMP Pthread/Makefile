CC = gcc
CXX = g++
LDLIBS = -lpng -lm
CFLAGS = -O3 -g3
CXXFLAGS = $(CFLAGS)

final_a: CFLAGS += -pthread -mavx512f -march=native
final_b: CC = mpicc
final_b: CXX = mpicxx
final_b: CFLAGS += -fopenmp -mavx512f -march=native
# 不再手動加入 -lmpi，mpicxx會自動鏈MPI庫

TARGETS = final_seq final_a final_b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
# module load icc openmpi
# ./judge.sh final_b t1000  