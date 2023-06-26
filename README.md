# Big Data Computing course

*(full text of each assignment is available in its folder)*

## Homework 1

You will develop a Spark program to analyze a dataset of an online retailer which contains several transactions made by customers, where a transaction represents several products purchased by a customer. Your program must be designed for very large datasets.

## Homework 2

Homeworks 2 and 3 will focus on the k-center with z outliers problem, that is, the robust version of the k-center problem which is useful in the analysis of noisy data (a quite common scenario in big data computing). Given a set P of points and two integers k and z, the problem requires to determine a set S ⊂ P of k centers which minimize the maximum distance of a point of P-Z from S, where Z are the z farthest points of P from S. In other words, with respect to the standard k-center problem, in the k-center with z outliers problem, we are allowed to disregard the z farthest points from S in the objective function. Unfortunately, the solution of this problem turns out much harder than the one of the standard k-center problem. The 3-approximation sequential algorithm by Charikar et al. for k-center with z outliers, which we call kcenterOUT, is simple to implement but has superlinear complexity (more than quadratic, unless sophisticated data structures are used).

The two homeworks will demonstrate that in this case a coreset-based approach can be successfully employed. In Homework 2 you will implement the 3-approximation sequential algorithm and will get a first-hand experience of its inefficiency. In Homework 3, you will implement a 2-round MapReduce coreset-based algorithm for the problem, where the use of the inefficient 3-approximation is confined to a small coreset computed in parallel through the efficient Farthest-First Traversal.

## Homework 3

In this homework, you will run a Spark program on the CloudVeneto cluster. The core of the Spark program will be the implementation of 2-round coreset-based MapReduce algorithm for k-center with z outliers, which works as follows: in Round 1, separately for each partition of the dataset, a weighted coreset of k+z+1 points is computed, where the weight assigned to a coreset point p is the number of points in the partition which are closest to p (ties broken arbitrarily); in Round 2, the L weighted coresets (one from each partition) are gathered into one weighted coreset of size (k+z+1)*L, and one reducer runs the sequential algorithm developed for Homework 2 (SeqWeightedOutliers) on this weighted coreset to extract the final solution. In the homework you will test the accuracy of the solution and the scalability of the various steps.