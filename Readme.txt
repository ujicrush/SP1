Here is a brief description of this repository:

The Functions_MCFND folder contains the code for three Benders decomposition algorithms for the Multi-commodity Capacitated Fixed-charge Network Design (MCFND) problem, based on the approach proposed in Bertsimas' paper. The three algorithms included are:

1. Stochastic Benders Decomposition - the main focus of this semester project, which aims to integrate this method with power grid optimization.
2. Traditional Parallel Single-Cut Benders Decomposition - a conventional approach that uses single-cut method and parallel computation.
3. Traditional Parallel Multi-Cut Benders Decomposition - another conventional approach that uses multi-cut method and parallel computation.

A multi-dimensional comparison of these three algorithms has been conducted across four benchmark instances, evaluating metrics such as computational time, number of iterations, and objective values. The comparison results can be found in the Comparison results.txt file.

The four benchmark instances are located in the **Benchmark** folder. For the two traditional parallel Benders decomposition algorithms, you need to copy the corresponding benchmark setup into the problem definition section at the beginning before running the code.

The code for the two traditional parallel Benders decomposition methods is located in the Traditional Benders Decomposition folder, while the main focus of this project, the new stochastic algorithm, can be found in the Stochastic Benders Decomposition folder.

[1]	D. Bertsimas, R. Cory-Wright, J. Pauphilet, and P. Petridis, ‘A Stochastic Benders Decomposition Scheme for Large-Scale Stochastic Network Design’, Inf. J. Comput., p. ijoc.2023.0074, Nov. 2024, doi: 10.1287/ijoc.2023.0074.
