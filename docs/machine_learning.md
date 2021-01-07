---
title: machine learning
permalink: /machine_learning/
---

# Machine Learning
## Linear Regression and Logistic Regression
### Linear Regression:
y = $w^Tx$

### Logitic Regression:
p(Y|X)  = sigmoid($w^Tx$)
Where the sigmoid is s(x) = $1/(1+e^-x)$
Threshold is typically p(Y|X) less or more than 0.5

## Clustering
### K-means: a iterative process to assign data points to k number of groups based on distance to the center of the groups. Different initiation could lead to different results.
* Assumes mixture weights are equal $\pi_k = 1/k$
* Assumes the covariance matrix are all $ \Sigma_k = \sigma*I $ for all clusters, which means each cluster has the same spherical structure
* There is no likelihood attached to K-means, which makes it harder to understand what assumptions we are making on the data.
* Each feature is treated equally, so the clusters produced by K-means will look spherical. We can also infer this by looking at the sum of squares in the objective function, which we have seen to be related to spherical Gaussians.
* Each cluster assignment in the optimization is a hard assignment - each point belongs in exactly one cluster. A soft assignment would assign each point to a distribution over the clusters, which can encode not only which cluster a point belongs to, but also how far it was from the other clusters.

### Expectation Maximization: an iterative process, but a "soft" verion of k-means. 

EM1: A soft assignment would assign each point to a distribution over the
clusters, which can encode not only which cluster a point belongs to, but also how far it was from the other clusters.
* Assumes mixture weights are equal $\pi_k = 1/k$
* Assumes the covariance matrix are all $ \Sigma_k = \sigma*I $ for all clusters, which means each cluster has the same spherical structure
* The only difference from K-means is that the assignments to cluster are soft (probabilistic), while K-means assignment is hard.

