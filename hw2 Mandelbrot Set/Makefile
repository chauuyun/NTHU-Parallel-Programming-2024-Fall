CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 -g3
hw2a: CFLAGS += -pthread -mavx512f -march=native
hw2b: CC = mpicc
hw2b: CXX = mpicxx
# hw2b: CFLAGS += -fopenmp
hw2b: CFLAGS += -fopenmp -lm -lpng -lmpi -mavx512f -march=native
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)