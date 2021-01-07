---
title: machine learning
permalink: /machine_learning/
---

# Machine Learning
## Linear Regression and Logistic Regression
### Linear Regression:
* y = $w^Tx$
* MLE
* Could add ridge penalizer: MAP

### Logitic Regression:
* p(Y|X)  = sigmoid($w^Tx$) transforms $w^Tx$ to a probability
* Where the sigmoid is s(x) = $1/(1+e^{-x})$
* Threshold is typically p(Y|X) less or more than 0.5
* MLE

I have written a logistic regression algrithm below:
```
# define helper functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def score(X, y ,w):
    return y *(X @ w)

def compute_loss(X, y, w):
    logloss = - sum(np.log(sigmoid(score(X, y, w))))/len(y)
    return logloss

def compute_gradients(X, y, w):
    dw = - sigmoid( - score(X, y, w)) * y @ X/len(y)
    return dw
    
def prediction(X, w):
    return sigmoid(X @ w)

def decision_boundary(prob):
    return 1 if prob >= .5 else -1

def classify(predictions, decision_boundary):
    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of -1s and 1s 
    '''
    db = np.vectorize(decision_boundary)
    return db(predictions).flatten()

def accuracy(X, y, w):
    y_pred = np.sign(X @ w)
    diff = y_pred - y
    # if diff is zero, then correct
    return 1 - float(np.count_nonzero(diff)) / len(diff)
    

# main - run model
for _ in range(50):
    ## Compute Loss
    loss_train = compute_loss(X_train, y_train, w)
    loss_test = compute_loss(X_test, y_test, w)

    ## Compute Gradients
    dw = compute_gradients(X_train, y_train, w)

    ## Update Parameters
    w = w - lr * dw

    ## Compute Accuracy and Loss on Test set (x_test, y_test)
    accuracy_train = accuracy(X_train, y_train, w)
    accuracy_test = accuracy(X_test, y_test, w)
    
    ## Save acc and loss
    accuracies_test.append(accuracy_test)
    accuracies_train.append(accuracy_train)
    losses_test.append(loss_test)
    losses_train.append(loss_train)

##Plot Loss and Accuracy
plt.plot(accuracies_test, label = "Test Accuracy")
plt.plot(accuracies_train, label = "Train Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title("Accuracies over Iterations")
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.show()

plt.plot(losses_test, label = "Test Loss")
plt.plot(losses_train, label = "Train Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title("Loss over Iterations")
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.show()

```

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

