# National Tsing Hua University 2024 Fall - CS542200 Parallel Programming

## HW1: Odd-Even Sort

In this assignment, you are required to implement the odd-even sort algorithm using **MPI**.

## HW2: Mandelbrot Set

In this assignment, you are asked to parallelize the sequential MandelbrotSet program using **Pthread**, **OpenMP**, and **MPI**.

## HW3: All-Pairs Shortest Path

In this assignment, you are asked to solve the all-pairs shortest path problem with CPU threads and then further accelerate the program with CUDA accompanied by Blocked Floyd-Warshall algorithm.

## HW4: Flash Attention

This assignment focuses on implementing the forward pass of the FlashAttention. You will gain an understanding of how efficient attention mechanisms work and how CUDA can be utilized to accelerate them. In this assignment, you will realize the performance benefits of optimizing the attention computation on GPUs. Finally, you are encouraged to explore and apply additional optimization strategies to maximize performance.

## Final Project: Dynamic Programming Minimum Cost to Cut a Stick 
[Leetcode 1547. Dynamic Programming Minimum Cost to Cut a Stick](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/description/)
 
This project, conducted by [Weber Lin](https://github.com/xswzaq44321) and I, focuses on optimizing and implementing the "Minimum Cost to Cut a Stick" problem using dynamic programming in a parallel programming context. The optimization process involves reducing the computational complexity of solving the problem, which calculates the minimum cost of cutting a stick at specified positions.

### Key implementations

- Dynamic Programming on CPU: Establishing a state transition relation to calculate the minimum cost for each cutting segment.
- CUDA-based GPU Optimization: Enhancing performance through techniques such as coalesced memory access, parallel reduction, and loop unrolling to achieve significant speedup.
- Hybrid MPI and OpenMP: Combining distributed and shared memory approaches to manage tasks efficiently across multiple processors.
Pthread Implementation: Parallelizing dynamic programming computations with threads to distribute workloads.

The optimizations achieved substantial speed improvements, reducing execution times from 73.30 seconds to 7.10 seconds for large test cases. The project demonstrates effective use of modern parallel programming techniques to solve computationally intensive problems. 
Additional details can be found in [this repository](https://github.com/chauuyun/Final-Project-Dynamic-Programming-Minimum-Cost-to-Cut-a-Stick).

## Grade

Overall Score: A+

HW1: 92.3

HW2: 91.11

HW3: 98.5

HW4: 99

Final Project: 93
