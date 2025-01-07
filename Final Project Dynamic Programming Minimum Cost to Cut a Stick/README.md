
# Final Project: Dynamic Programming Minimum Cost to Cut a Stick 

[Leetcode 1547. Dynamic Programming Minimum Cost to Cut a Stick](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/description/)
 
## Project Overview

The project, led by by [Weber Lin](https://github.com/xswzaq44321) and I, tackles the classic "Minimum Cost to Cut a Stick" problem through an optimized implementation that harnesses dynamic programming within a parallel computing framework. The objective was to minimize the computational overhead of determining the least expensive sequence of cuts on a stick, given a set of predefined cutting positions.

## Key Methodologies and Implementations

- Dynamic Programming on CPU
The foundation of the solution is built on dynamic programming. By formulating a state transition relation, the algorithm efficiently computes the minimum cost for every possible segment of the stick. This methodical breakdown ensures that all potential subproblems are solved optimally, building up to the solution of the overall problem.

- CUDA-based GPU Optimization
To substantially accelerate computation, the project harnessed GPU capabilities via CUDA. Key techniques employed in this phase include:

- Coalesced Memory Access: Optimizing memory access patterns to ensure that threads within a warp access contiguous memory locations, reducing latency.

- Parallel Reduction: Efficiently combining results from multiple threads to reduce a data set to a single value, crucial for aggregating partial costs.

- Loop Unrolling: Minimizing loop overhead and increasing instruction-level parallelism by expanding loops manually.
These optimizations collectively delivered significant performance gains by fully exploiting the parallel architecture of GPUs.

- Hybrid MPI and OpenMP Integration
Recognizing the potential of distributed and shared memory paradigms, the project combined MPI (Message Passing Interface) and OpenMP.

MPI was employed to distribute independent tasks across different processors in a cluster, ensuring scalability over multiple nodes.
OpenMP provided shared memory parallelism within each node, further dividing tasks among processor cores.
This hybrid approach led to a more balanced workload distribution and efficient resource utilization across a heterogeneous computing environment.

- Pthread Implementation
Complementing the CUDA and MPI/OpenMP strategies, the team also implemented a solution using POSIX threads (Pthreads). By decomposing the dynamic programming task into concurrently executable threads, the program distributed workloads across multiple cores on a single machine, maximizing throughput and reducing idle time.

## Results and Impact

The comprehensive suite of optimizations delivered dramatic improvements in execution speed. For large test cases, the execution time plummeted from 73.30 seconds in a naive, sequential setup to just 7.10 seconds after applying parallelization and algorithmic optimizations.

This achievement not only underscores the power of modern parallel programming paradigms but also demonstrates a practical roadmap for tackling computationally intensive problems. By systematically analyzing bottlenecks and applying targeted optimizations—from dynamic programming refinements on the CPU to leveraging GPU acceleration, distributed computing, and multithreading—the project sets a benchmark for future work in performance-critical applications.

Overall, the collaborative effort combined theoretical insights with practical engineering to deliver a robust, high-performance solution to a traditionally challenging problem.