---
title: machine learning
permalink: /machine_learning/
---

# Machine Learning
## Clustering
K-means: a iterative process to assign data points to k number of groups based on distance to the center of the groups. Different initiation could lead to different results.
* assumes mixture weights are equal
* assumes the covariance matrix are all $'\sigma*I'$, which means each cluster has the same spherical structure

Expectation Maximization: an iterative process, but a "soft" verion of k-means. The EM for Gaussian mixtures with $' \Sigma_k = \sigma*I$ and $\pi_k = 1/k '$ also assumes the same with the K-means.
However, the only difference from K-means is that the assignments to cluster are soft (probabilistic), while K-means assignment is hard.

