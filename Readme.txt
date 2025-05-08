以下对此repository进行简要说明:

在Functions_MCFND folder中包含了以Bertsimas的文章为基准的Multi-commodity Capacitated Fixed-charge Network Design（MCFND）问题下的三种Benders decomposition algorithms的代码。三种algorithm分别是：bertsimas提出的stochastic benders decomposition(也就是本次semester project试图与电网融合的)，传统的并行运算的single-cut benders decomposition，传统的并行运算的multi-cut benders decomposition。

对这三种算法在四种算例下进行了多维度的比较，包括他们的运算时间，运算iteration数量，目标值等。对比的结果在Comparison results.txt文件中。

四种算例在Benchmark folder中，在传统的两种算法的py文件运行时，需要复制对应的benchmark在开头问题设置部分进去替换。

两种传统的并行运算benders decomposition代码在Traditional Benders Decomposition文件夹中，我们主要研究的新型算法在Stochastic Benders Decomposition文件夹中。

[1]	D. Bertsimas, R. Cory-Wright, J. Pauphilet, and P. Petridis, ‘A Stochastic Benders Decomposition Scheme for Large-Scale Stochastic Network Design’, Inf. J. Comput., p. ijoc.2023.0074, Nov. 2024, doi: 10.1287/ijoc.2023.0074.
