---
title: machine learning
permalink: /machine_learning/
---

# Machine Learning
## General Supervised Learning Process (Supervised if label exists)
* Model: usually refers to the mathematical structure of by which the prediction 𝑦𝑖 is made from the input 𝑥𝑖.
   * Parameters (if any) and hyper-parameters: the parameters of the model that needs to be trained/optimized to obtain the best prediction. Can be optimized using MLE or MAP. Some non-parametric models like KNN or Decision Tree does not have parameters, but have hypter-parameters like number of neighbors or tree depth, which is pre-determined can be optimized through cross validation. 
* Objective function: training loss + regularization
  * regularization typically reduces the "strength" of the weights or parameters to prevent overfitting by adding an additional term using weights itself in the loss function
* Optimization using testing and validation data
   * Optimize the model by changing the parameters to achieve the lowest objective function and thus the best prediction
   * Minimize the objective loss function: closed-form solution, gradient descent
      * Batch gradient descent: could compute all the training set together at once, could compute subset groups of the data and get the avg gradient to update, but the underlying idea is for one epoch (finished going through all training set), there is only one weight update.
      * Mini-Batch gradient descent: update weights with a subset of sample (a mini-batch); does not wait until gradient for all data are computed. For one epoch, there are several updates to the weight. When the mini-batch size = 1, this is stochastic gradient descent, which could be unstable.
   * A simple decision tree achieves a similar goal by maximizing information gain
* Evaluation metrics using testing data (accuracy, f-score, AUC) 

([source](https://xgboost.readthedocs.io/en/latest/tutorials/model.html))

## General Unsupervised Learning Process (Unsupervised if label does not exist)
* Typically the output is a variation of the input
* Can help to reduce dimensions/perform feature extraction (auto-encoders)
* Can help to create labels


## Generative vs. Discriminative (Typically applicable for classifiers)
* Generative: Form a probability distribution p(X, Y) over the input X (which we treat as a random vector) and label Y (which we treat as a random variable), 
  and we classify an arbitrary datapoint x with the class label that maximizes the joint probability.
    * p(X,Y) is learned from p(Y) and p(X|Y_k) for each class k. 
    * Uses Bayes' rule to learn p(Y|X) = p(X|Y) * p(Y)/P(X). P(Y) is the prior probability.
    * Flexible, quick to learn, and can generate new samples
* Discriminative: learn p(Y|X) directly.

## Linear Regression and Logistic Regression
### Linear Regression:
* y = w^Tx
* Four assumptions, see statistics section
* No regularization: MLE to find the best parameter
* With regularization: MAP to find the best parameter
* Loss Function: base is mean squared error = mean(|y_pred - y_actual|^2) or regression squared error = sum(|y_pred - y_actual|^2)
    * L2 regularizer: uses (L2 norm of the weights)^2 as regularizer, thus penalizes larger weights
    * L1 regularizer: uses (L1 norm of the weights)^1 as regularizer, thus penalizes larger weights a little bit but also push smaller weights to 0
    * L0 regularizer: counts of non-zero weights (not convex), thus penalizes smaller weights to 0; no closed form solution, requires searching
    * elastic net: L1 + L2 loss

### Logistic Regression:
* p(Y|X)  = sigmoid(w^Tx) transforms w^Tx to a probability
* The sigmoid activation function is s(x) = 1/(1+e^{-x})
* Threshold is typically p(Y|X) less or more than 0.5
* No regularization: MLE to find the best parameter
* With regularization: MAP to find the best parameter
* Loss function: cross entropy loss (this loss does not have closed-form solution, so we need to do gradient descent)
   * can also add regularization
* In general, we can have multiple predictor variables in a logistic regression model. 
* logit(p) = log(odds) = log(p/q) = a + bX, where q=1-b for binary classification
* b = log(odds_p) – log(odds_q)
* Logistic regression is not great for multi-class classification. Try LDA/QDA instead.

I have written a logistic regression algorithm below:
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

We can tell from this algorithm that logistic regression is also a linear model, and it is just applying the sigmoid activation/softmax function on y=Xw to obtain a probability distribution for classification purposes. 
It is considered linear because the prediction or outcome depends on the linear combination of the features.

## LDA and QDA
* Estimate p(Y|X) indirectly using the Bayes Rule: p(Y|X) = p(X,Y)/p(X) = p(X|Y) * p(Y)/p(X).
* Do not require the features to be independent. (Naive Bayes assumes features are independent: given Y, X(1), X(2)...X(d) are independent.)
* LDA assumes p(X=x|Y=k) = k ~ N(mu_k, Sigma); Normal distribution's parameters mu_k and sigma are estimated from training data
    * LDA has k*p + p(p+1)/2 parameters
* QDA assumes p(X=x|Y=k) = k ~ N(mu_k, Sigma_k); Normal distribution's parameters mu_k and sigma_k are estimated from training data
    * LDA has k*p + kp(p+1)/2 parameters

## Perceptron and SVM
### Perceptron
* Uses stochastic gradient descent.
* The only hyper-parameter is the number of iterations.
* Prediction is sign(w^Tx), where label y takes value in -1 and 1 accordingly as well.
* For each data, if the prediction is correct (sign(w^Tx) = y), do not update the weights; if the prediction is incorrect (sign(w^Tx)!=y), update the gradient and update the weights
    * If the prediction is incorrect, each update in weights essentially pulls the weight vector closer to the wrongly predicted data point (vector addition).
  The weight update formula is: w = w' + learning_rate * correctly_predicted_or_not * x
    * learning_rate = 1/2 typically, but the algorithm will converge for 0<learning_rate<the largest eigenvalue of the X^TX matrix/largest [singular values of X squared] if data is linearly separable. Convergence rate is proportional to min(eigenvalues of X^TX)/max(eigenvalues of X^TX).
    * correctly_predicted_or_not is calculated simply by y - sign(w^Tx). The value is 0 when predicted correctly, thus w = w'; the value is 2 (y=1, prediction=-1) or -2 (y=-1, prediction=1), when predicted incorrectly. More specifically, push the normal vector in the direction of the mistake if
it was positively labeled and away if it was negatively labeled. Setting learning_rate = 1/2 will simplify the weight update formula to w = w' + correctly_prediction_or_not * x.
* Fast but not stable, since computing the gradient using only one data point is fast, but each data point could change the prediction. If it’s possible to separate the data with a hyperplane (i.e. if it’s linearly separable), then the algorithm will converge to that hyperplane. If the data is not separable then perceptron is unstable and bounces around. 
* Could use voted perceptron algorithm or average perceptron algorithm to solve some of the stability issue.
    * Voted perceptron algorithm need to save/memorize all the previous predictions to make a vote.
    * Averaged perceptron algorithm does not need to save/memorize all the previous predictions. We can also use kernel with this algorithm.
* Number of mistakes in perceptron algorithm has a upper bound  M = R^2/gamma^2, where R = max||X||_2 (size of the largest X), and gamma (the margin) < y w*^Tx, where the star means optimal w. Gamma is the margin and >0 if the data is separable.
* Other variations:
    * Passive-aggressive perceptron model (Margin-Infused Relaxed Algorithm): uses hinge loss. L = 0 if yw^Tx >=1, else 1 - yw^Tx. Multi-class.
* Unsolvable issue:
    * Will have A solution, but not necessarily a 'good' separator. The perceptron algorithm is affected by the order the data is processed. There are infinitely many separating planes that can be drawn using this algorithm to separate the data. Therefore, the development of SVM is a better algorithm.

([source 1](http://web.mit.edu/6.S097/www/resources/L01.pdf))
([source 2](https://www.seas.upenn.edu/~cis520/lectures/15_online_learning.pdf))
 

### SVM
* Pick the best linear classifier with the largest margin (solves the perceptron problem)
* Separable case 
    * Choose w^Tx + b = +-1 for the positive and negative margins
    * The margin is 2/||w||
    * Cost function is to 1) maximize margin 2/||w|| subject to w^Tx + b >=1 if y=1 and w^Tx + b <=-1 if y=-1, 
      or 2) minimize ||w||^2 subject to y(w^Tx+b)>=1 for i = 1...N
    * This is a quadratic optimization problem subject to linear constraints and there is a unique minimum
    * Can just use hard margin SVM.
    * Uses 0-infinity loss.
    * Scale invariant.
* Unseparable case uses hinge loss.
    * Use soft margin SVM.
    * Uses hinge loss to approximate 0-1 loss. (Initially, soft margin version is 0-1 loss, but the problem is that it is not convex.)
    * Slack variable_i is max(1-y(w_i^Tx_i),0).
    * w*_soft = argmin C*sum(max(1-y(w_i^Tx_i),0)) + 1/2 ||w||^2
        * the first part of the loss function is the hinge loss, and the second part is the l2 regularizer (=w^Tw)
        * C in the first part is a regularization parameter: 
            * small C allows constraints to be easily ignored → large margin
            * large C makes constraints hard to ignore → narrow margin
            * when C is infinity, forces slack variable to be zero, thus enforces all constraints → hard margin
    * This is still a quadratic optimization problem and there is a unique minimum. Can be solved using gradient descent.
* SVM can be expressed as a sum over the support vectors. Only the support vectors are affecting the decision boundary/surface.
  

([source](http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/SupportVectorMachines.pdf))
([source](https://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf))

### Logistic Regression vs SVM
* Logistic regression focuses on maximizing the probability of the data. The farther the data lies from the separating hyperplane 
(on the correct side), the happier LR is.

* An SVM tries to find the separating hyperplane that maximizes the distance of the closest points to the margin (the support vectors). 
  If a point is not a support vector, it doesn’t really matter. 
  
* We could derive SVM by playing around with the likelihood ratio p(y=1|x)/p(y=0|x)

* Differ only that one uses logistic loss (cross entropy loss) and the other uses 1-infinity for the separable case and hinge loss for 
unseparable case.
  
* Logistic regression is more sensitive to outliers.

([source](http://www.cs.toronto.edu/~kswersky/wp-content/uploads/svm_vs_lr.pdf))
([source](https://gdcoder.com/support-vector-machine-vs-logistic-regression/))


## Trees
* Can help with feature importance and feature selection.
* Assumes hierarchy structure in the features.

### Decision tree: 
* Each level splits the samples based on each feature, and select feature order by maximizing information gain/mutual information (=difference in entropy and conditional entropy: H(Y) - H(Y|X))
  * Other ways to access split quality includes Gini impurity
* Uses information gain, so do not really have a loss function. The misclassification rate (loss function) is not sensible in this case, because different types of splits,
can produce the same misclassification rate, although one split is better than the other. See example in the source.
* Scales with log(n) for best case, and p for worst case; n = number of samples, p = number of features
* Easy to overfit; need ways to regularize the model (early stopping and pruning)
    * Early Stopping criteria:
        * Limited depth: don’t split if the node is beyond some fixed depth depth in the tree
        * Node purity: don’t split if the proportion of training points in some class is sufficiently high
        * Information gain criteria: don’t split if the gained information/purity is sufficiently close to zero
    * Pruning: prune a fully-grown tree by re-combining splits if doing so reduces validation error
  
  
（[source](https://www.eecs189.org/static/notes/n25.pdf))

### Random Forest
* An ensemble model. Random Forests grows many classification trees. To classify a new object from an input vector, put the input vector down each of the trees in the forest. 
  Each tree gives a classification, and we say the tree "votes" for that class. The forest chooses the classification having the most votes (over all the trees in the forest).
* Uses bagging (subset of sample size, usually 2/3 of the data, with replacement) and feature randomization (subset of features, usually sqrt(M), with replacement) as weak learners; 
  Two ways to regularize the model to prevent the tree based model to overfit
* Usually k = 1,000 trees
* Both the size of the random subsample of training points and the number of features at each split are hyperparameters which should be tuned through cross-validation.
* Out-of-bag (oob) data: When the training set for the current tree is drawn by sampling with replacement, about one-third of the cases are left out of the sample.
  This oob (out-of-bag) data is used to get a running unbiased estimate of the classification error as trees are added to the forest. 
  It is also used to get estimates of variable importance.
* Proximities: After each tree is built, all of the data are run down the tree, and proximities are computed for each pair of cases. If two cases occupy the same terminal node, 
  their proximity is increased by one. At the end of the run, the proximities are normalized by dividing by the number of trees. 
  Proximities are used in replacing missing data, locating outliers, and producing illuminating low-dimensional views of the data.

Each tree is grown as follows:
* If the number of cases in the training set is N, sample N cases at random - but **with replacement**, from the original data. This sample will be the training set for growing the tree.
* If there are M input variables, a number m<<M is specified such that at each node, m variables are selected at random out of the M and the best split on these m is used to split the node. 
  The value of m is held constant during the forest growing.
* Each tree is grown to the largest extent possible. There is no pruning.

In the original paper on random forests, it was shown that the forest error rate depends on two things: correlation between trees, and strength of individual trees.
* The **correlation** between any two trees in the forest. Increasing the correlation increases the forest error rate.
* The **strength** of each individual tree in the forest. A tree with a low error rate is a strong classifier. Increasing the strength of the individual trees decreases the forest error rate.
* Reducing m reduces both the correlation and the strength. Increasing it increases both. Somewhere in between is an "optimal" range of m - usually quite wide. 
Using the oob error rate (see below) a value of m in the range can quickly be found. This is the only adjustable parameter to which random forests is somewhat sensitive.

Features and Remarks of Random Forests
* It is unexcelled in accuracy among current algorithms.
* It runs efficiently on large data bases.
* It can handle thousands of input variables without variable deletion.
* It gives estimates of what variables are important in the classification.
* It generates an internal unbiased estimate of the generalization error as the forest building progresses.
* It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing.
* It has methods for balancing error in class population unbalanced data sets.
* Generated forests can be saved for future use on other data.
* Prototypes are computed that give information about the relation between the variables and the classification.
* It computes proximities between pairs of cases that can be used in clustering, locating outliers, or (by scaling) give interesting views of the data.
* The capabilities of the above can be extended to unlabeled data, leading to unsupervised clustering, data views and outlier detection.
* It offers an experimental method for detecting variable interactions.
* Random forests does not overfit. You can run as many trees as you want. It is fast. Running on a data set with 50,000 cases and 100 variables, 
  it produced 100 trees in 11 minutes on a 800Mhz machine. For large data sets the major memory requirement is the storage of the data itself, 
  and three integer arrays with the same dimensions as the data. If proximities are calculated, storage requirements grow as the number of cases 
  times the number of trees.
  
Variable Importance using Random Forest:
* In every tree in the forest, apply the tree classification to the oob/validation data and count, and then count the number of correct predictions (A).
* Permute the values of the variable m in the oob/validation data and then apply every tree again. Count the number of correct predictions (B).
* Avg(A-B) across every tree is the raw score for feature importance.

([source](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm))

### Boosting:
Random forests treat each member of the forest equally, taking a plurality vote or an average over their outputs.
Boosting aims to combine the models (typically called weak learners in this context) in a more
principled manner. 

**The key idea is as follows: to improve our combined model, we should focus
on finding learners that correctly predict the points which the overall boosted model is currently
predicting inaccurately.** Boosting algorithms implement this idea by associating a weight with each
training point and iteratively reweighting so that mispredicted points have relatively high weights.
Intuitively, some points are “harder” to predict than others, so the algorithm should focus its efforts
on those.

#### Gradient Boosting: Adaboost
* A method for binary classification
* Weight each data differently based on predicted correctly or not. If predicted correctly, less weights; If predicted incorrectly, more weights.
    * How do we train a classifier if we want to weight different samples differently? One common way to do this is to resample from the original training set 
      every iteration to create a new training set that is fed to the next classifier. Specifically, we create a training set of size n by sampling n values from the original 
      training data with replacement, according to the distribution wi. (This is why we might renormalize the weights in step (d).) This
      way, data points with large values of wi are more likely to be included in this training set, and the
      next classifier will place higher priority on such data points.
* Compute weighted error (should be at most 0.5, otherwise model is even worse than random guessing).
* Re-weight the training points: if classified correctly, will decrease weight; otherwise, will increase weight. See formula in the source below.
* Uses exponential loss: L(y, y_hat) = e^{-y*y_hat}

#### Gradient Boosting: General
* AdaBoost assumes a particular loss function, the exponential loss function. Gradient boosting is a more general technique that allows an arbitrary differentiable loss function L(y, yˆ).
* The weak learner G could be a regressor in addition to classifier
* Matching pursuit (also applies to Adaboost): This means the algorithm will follow the residual; our overall predictor is an additive combination of pieces which are selected one-by-one in a greedy fashion. 
  The algorithm keeps track of residual prediction errors, chooses the “direction” to move based on these, and then performs a sort of line search to determine how far along that direction to move

([source](https://www.eecs189.org/static/notes/n26.pdf))
([source](https://alliance.seas.upenn.edu/~cis520/dynamic/2020/wiki/index.php?n=Lectures.Boosting))


## Clustering
* Properties of a good cluster: high inter-similarity; low intra-similarity

### K-means: an iterative process to assign data points to k number of groups based on distance to the center of the groups. Different initiation could lead to different results.
* Assumes cluster weights are equal $\pi_k = 1/k$
* Assumes the covariance matrix are all $ \Sigma_k = \sigma*I $ for all clusters, which means each cluster has the same spherical structure
* There is no likelihood attached to K-means, which makes it harder to understand what assumptions we are making on the data.
* Each feature is treated equally, so the clusters produced by K-means will look spherical. We can also infer this by looking at the sum of squares in the objective function, which we have seen to be related to spherical Gaussians.
* Each cluster assignment in the optimization is a hard assignment - each point belongs in exactly one cluster. A soft assignment would assign each point to a distribution over the clusters, 
  which can encode not only which cluster a point belongs to, but also how far it was from the other clusters.

### Soft K-means
* A soft assignment would assign each point to a distribution over the clusters, which can encode not only which cluster a point belongs to, but also how far it was from the other clusters.
* z= -β||x-c_k||^2. β is a tunable parameter indicating the level of “softness” desired
* This is now a weighted average of the xi - the weights reflect how much we believe each data point belongs to a particular cluster, and because we are using this information, our algorithm should not
  jump around between clusters, resulting in better convergence speed.
* Assumes the covariance matrix are all $ \Sigma_k = \sigma*I $ for all clusters, which means each cluster has the same spherical structure.
  
### Gaussian Mixture model
* Allow covariance matrix and mean of the clusters to be arbitrary
* Draw a value z from some distribution on the set of indices {1,...,K}. Draw a points X from Gaussian distribution N(mu_z, Sigma_z). mu and Sigma are different for each z.
* An example of latent variable model.
* If we have fit a MoG model to data (ie. we have determined values for µk, Σk, and the prior on z), then to perform clustering, we can use Bayes’ rule to determine the posterior P(z = k|x) and
assign x to the cluster k that maximizes this quantity. In fact, this is exactly our decision rule with QDA using a prior - the difference is that QDA, a supervised method, is given labels to fit the
mixture model, while in the unsupervised clustering setting we must fit the mixture model without the aid of labels. When Σk are not multiples of the identity, we can obtain non-spherical clusters,
which was not possible with K-means.
* Log likelihood contains both p(zi|xi;θ) and p(zi=k;theta)
    * When we perform QDA, we know the zi are known, deterministic quantities and thus the likelihood is simplified to Log likelihood  = sum(log(p(xi|zi;theta)))
    * Maximizing this is equivalent to fitting the individual class-conditional Gaussians via maximum likelihood, which is consistent with how we have described QDA in the past. 
    
### Both uses an **EM algorithm**
* Used to compute the MLE for latent variable models.
* Can also be used to impute missing data.
* EM for K-means:
    * Expectation (soft imputation): For each data point, compute a soft assignment ri(k) to the clusters - that is, a probability distribution over clusters. The soft assignment is obtained by using a softmax.
    * Maximization (parameter estimation): Update the centroids in an optimal way given the soft assignments computed in the first step. The resulting updates are a weighted average of the data points.
* EM for Gaussian Mixtures:
    * Expectation: Using the current parameter estimates, estimate p(zi|xi;θ). That is, perform soft imputation of the latent cluster variable.
    * Maximization: Estimate the parameters via MLE, using the estimates of p(zi|xi;θ) to make the computation tractable.
* EM for missing data:
    * Expectation: Soft imputation of the data - fill in the missing data (“impute”) with a probability distribution over all its possible values (“soft”).
    * Maximization: Parameter updates given the imputed data.
    
([source](https://www.eecs189.org/static/notes/n19.pdf))

## Class Imbalance Problem
1. Use alternative evaluations: use f-score
2. Sampling the data (oversampling the small class, and undersampling the large class)
3. Use tree based models

## Bayes Belief Net
([source](http://www.cs.cmu.edu/~mgormley/courses/10601bd-f18/slides/lecture21-bayesnet.pdf))

## Hidden Markov Model, RNN
### HMM
* Assumes Markov Property, and assumes independent assumption
    * Estimate transition matrix and emission matrix
    * History is forgotten with an exponential decay
  
### RNN
* input x, goes through hidden state s, output an observation (could be the probability of the next word)
* Softmax s(z) transforms the K-dimensional real valued output z to a distribution o

([source](https://www.seas.upenn.edu/~cis520/lectures/28_recurrent_networks.pdf))

## Neural Networks
* Input layers (number of neurons = number of features), hidden layers, and output layers
* If no activation functions, then it is basically a linear function

## Multi-class Neural Networks Classification
([source](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/video-lecture))

### Gradient Vanishing and Explosion in NN/RNN
* Due to long backpropagation process, the gradient could get smaller and smaller, or larger and larger in the chain rule process.
    * The forward pass has nonlinear activation functions which squash the activations, preventing them from blowing up.
    * The backward pass is linear, so it’s hard to keep things stable. There’s a thin line between exploding and vanishing.
* Solution includes:
    * Gradient clipping: we prevent gradients from blowing up by rescaling them so that their norm is at most a particular value η
    * Input reversal (application: RNN for machine translation)
    * Identity Initialization of the weight matrix with ReLU as activation functions
        * Negative activations are clipped to zero, and for the positive activations, units simply retain their value in absence of inputs
    * Long-Term Short Term Memory: a type of gated recurrent neural network - input gate, output gate, forget gate
        * Replace each single unit in an RNN by a memory block
  
([source](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf))
  
### Overfitting in NN
* Neural networks are easy to overfit
* Solutions includes:
    * Adding L2 regularization to the cost function
    * Add dropouts:  during training, we will randomly “drop” with some probability (1 − p) a subset of neurons during each forward/backward pass (or equivalently, 
      we will keep alive each neuron with a probability p). One intuitive reason why this technique should be so effective is that what dropout is doing is
      essentially doing is training exponentially many smaller networks at once and averaging over their predictions.
      
### Activation Functions in NN
* Sigmoid
* Tanh: The tanh function is an alternative to the sigmoid function that is often found to converge faster in practice. The primary difference between tanh and sigmoid is that tanh output ranges from −1 to
1 while the sigmoid ranges from 0 to 1.
* Hard tanh: The hard tanh function is sometimes preferred over the tanh function since it is computationally cheaper. It does however
saturate for magnitudes of z greater than 1.
* ReLU: The ReLU (Rectified Linear Unit) function is a popular choice of activation since it does not saturate even for larger values of z and
has found much success in computer vision applications
* Leaky ReLU: Traditional ReLU units by design do not propagate any error for non-positive z – the leaky ReLU modifies this such
that a small error is allowed to propagate backwards even when z is negative


([source](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes03-neuralnets.pdf))
