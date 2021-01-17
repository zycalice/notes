---
title: machine learning
permalink: /machine_learning/
---

# Machine Learning
## General Supervised Learning Process (Supervised if label exists)
* Model: usually refers to the mathematical structure of by which the prediction 𝑦𝑖 is made from the input 𝑥𝑖.
   * Parameters (if any) and hyper-parameters: the parameters of the model that needs to be trained/optimized to obtain the best prediction. Can be optimized using MLE or MAP. Some non-parametric models like KNN or Decision Tree does not have parameters, but have hypter-parameters like number of neighbors or tree depth, which is pre-determined can be optimized through cross validation. 
* Objective function: training loss + regularization
  * regularization typically reduces the "strength" of the weights or parameters to prevent overfitting by additing an additional term using weights itself in the loss function
* Optimization using testing and validation data
   * Optimize the model by changing the parameters to achieve the lowest objective function and thus the best prediction
   * Minimize the objective loss function: closed-form solution, gradient descent
      * Batch gradient descent: could compute all the training set togther at once, could compute subset groups of the data and get the avg gradient to update, but the underlying idea is for one epoch (finished going through all training set), there is only one weight update.
      * Mini-Batch gradient descent: update weights with a subset of sample (a mini-batch); does not wait until gradient for all data are computed. For one epoch, there are several updates to the weight. When the mini-batch size = 1, this is stochastic gradient descent, which could be unstable.
   * A simple decision tree achieves a similar goal by maximizing information gain
* Evaluation metrics using testing data (accuracy, f-score, AUC) 

([reference](https://xgboost.readthedocs.io/en/latest/tutorials/model.html))

## General Unsupervised Learning Process (Unsupervised if label does not exist)
* Typically the output is a variation of the input
* Can help to reduce dimensions/perform feature extraction (auto-encoders)
* Can help to create labels

## Linear Regression and Logistic Regression
### Linear Regression:
* y = $w^Tx$
* Ordinary least squared assumes y is normally distributed
* No regularization: MLE to find the best parameter
* With regularization: MAP to find the best parameter
* Loss Function: base is mean squared error = $mean(|y_pred - y_actual|^2)$ or regression squared error = $sum(|y_pred - y_actual|^2)$
    * L2 regularizer: uses (L2 norm of the weights)^2 as regularizer, thus penalizes larger weights
    * L1 regularizer: uses (L1 norm of the weights)^1 as regularizer, thus penalizes larger weights a little bit but also push smaller weights to 0
    * L0 rgularizer: counts of non-zero weights (not convex), thus penalizes smaller weights to 0; no closed form solution, requires searching
    * elastic net: L1 + L2 loss

### Logistic Regression:
* $p(Y | X)$  = sigmoid($w^Tx$) transforms $w^Tx$ to a probability
* Where the sigmoid is s(x) = $1/(1+e^{-x})$
* Threshold is typically $p(Y | X)$ less or more than 0.5
* MLE
* Loss function: cross entropy loss (this loss does not have closed-form solution, so we need to do gradient descent)
   * can also add regularization
* In general, we can have multiple predictor variables in a logistic regression model. The cofficients are the log odds of p/(1-p)

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
In the above algorithm, number of iterations is pre-defined. One could also set an error threshold to determine when the weight updates should stop.

We can also tell from this algorithm that logistic is also a linear model, and it is just applying the sigmoid activation/softmax function on y=Xw to optain a probability distribution for classification purposes. It is considered linear because the prediction or outcome depends on the linear combination of the features.

## Perceptron and SVM
### Perceptron
* Uses stochastic gradient descent.
* The only hyper-parameter is the number of iterations.
* Prediction is $sign(w^Tx)$, where label y takes value in -1 and 1 accordingly as well.
* For each data, if the prediction is correct $(sign(w^Tx) = y)$, do not update the weights; if the prediction is incorrect $sign(w^Tx)!=y$, update the gradient and update the weights
    * If the prediction is incorrect, each update in weights essentially pulls the weight vector closer to the wrongly predicted data point (vector addition).
  The weight update formula is: W = W' + learning_rate * correctly_predicted_or_not * x
    * learning_rate = 1/2 typically, but the algorithm will converge for 0<learning_rate<largest eigenvalue of the $X^TX$ matrix/largest [singular values of X squared] if data is linearly separable. Convergence rate is proportional to min(eigenvalues of $X^TX$)/max(eigenvalues of $X^TX$).
    * correctly_predicted_or_not is calculated simply by y - $sign(w^Tx)$. The value is 0 when predicted correctly, thus W = W'; the value is 2 (y=1, prediction=-1) or -2 (y=-1, prediction=1), when predicted incorrectly. More specifically, push the normal vector in the direction of the mistake if
it was positively labeled and away if it was negatively labeled. Setting learning_rate = 1/2 will simplify the weight update formula to W = W' + correctly_prediction_or_not * x.
* Fast but not stable, since computing the gradient using only one data point is fast, but each data point could change the prediction. If it’s possible to separate the data with a hyperplane (i.e. if it’s linearly separable), then the algorithm will converge to that hyperplane. If the data is not separable then perceptron is unstable and bounces around. 
* Could use voted perceptron or average perceptron to solve some of the stability issue.
    * Voted perceptron need to save/memorize all the previous predictions to make a vote.
    * Averaged percetron does not need to save/memorize all the previous predictions. We can also use kernal with this algorithm.
* Number of mistakes in perceptron algorithm has a upper bound  M = $R^2/gamma^2$, where R = $max||X||_2$ (size of the largest X), and gamma (the margin) $< y W^{*T}X$. Gamma is the margin and >0 if the data is separable.
* Other variations:
    * Passive-agressive perceptron model (Margin-Infused Relaxed Algorithm): uses hinge loss. L = 0 if $yw^Tx >=1$, else $1 - yw^Tx$.
* Unsolvable issue:
    * Will have A solution, but not necessarily a 'good' separator. The perceptron algorithm is affected by the order the data is processed. There are infinitely many separating planes that can be drawn using this algorithm to separate the data. Therefore, the development of SVM is a better algorithm. 
    
（[source 1](http://web.mit.edu/6.S097/www/resources/L01.pdf) ｜ [source 2](https://www.seas.upenn.edu/~cis520/lectures/15_online_learning.pdf)）
 

### SVM
* Separabale case uses 0-inifity loss.
* Unseparatebl case usese hinge loss.
（[source](http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/SupportVectorMachines.pdf)）


## Trees
### Decision tree: 
* Each level splits the samples based on each feature, and select feature order by maximizing information gain (difference in entropy)
* Scales with log(n) for best case, and p for worst case; n = number of samples, p = number of features
* Assumes a hierchy structure
* Easy to overfit; need ways to regularize the model

### Randome Forest
* An ensemble model
* Uses bagging (subset of n) and boosting (subset of features) as weak learners
* Two ways to regularize the model to prevent the tree based model to overfit

### Gradient Tree Boosting
* 


## Clustering
### K-means: an iterative process to assign data points to k number of groups based on distance to the center of the groups. Different initiation could lead to different results.
* Assumes cluster weights are equal $\pi_k = 1/k$
* Assumes the covariance matrix are all $ \Sigma_k = \sigma*I $ for all clusters, which means each cluster has the same spherical structure
* There is no likelihood attached to K-means, which makes it harder to understand what assumptions we are making on the data.
* Each feature is treated equally, so the clusters produced by K-means will look spherical. We can also infer this by looking at the sum of squares in the objective function, which we have seen to be related to spherical Gaussians.
* Each cluster assignment in the optimization is a hard assignment - each point belongs in exactly one cluster. A soft assignment would assign each point to a distribution over the clusters, which can encode not only which cluster a point belongs to, but also how far it was from the other clusters.

### Guassian Mixture model: also an iterative process to cluster, but a "soft" verion of k-means. A soft assignment would assign each point to a distribution over the clusters, which can encode not only which cluster a point belongs to, but also how far it was from the other clusters.
* Assumes cluster weights are equal $\pi_k = 1/k$
* Assumes the covariance matrix are all $ \Sigma_k = \sigma*I $ for all clusters, which means each cluster has the same spherical structure
* The only difference from K-means is that the assignments to cluster are soft (probabilistic), while K-means assignment is hard.


### Both uses an **EM algorithm**

([source](https://www.eecs189.org/static/notes/n19.pdf))
