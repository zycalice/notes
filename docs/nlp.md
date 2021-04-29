---
title: natural language processing
permalink: /nlp/
---
# Natural Language Processing (Statistical Learning)
# Bag of Words and Naive Bayes

# Natural Language Processing (Deep Learning)
## Word Vectors
### 1. One Hot Encoders
The simplest form of word vectors; but it is not feasible to find any similarity between the vectors.

### 2. SVD Based Methods
For this class of methods to find word embeddings (otherwise known as word vectors), we first loop over a massive dataset and accumulate word co-occurrence counts in some form of a matrix X, 
and then perform Singular Value Decomposition on X to get a USV^T decomposition. We then use the rows of U as the word embeddings for all words in our dictionary. 

#### 2.1 Word-Document Matrix
* Essentially each row is a word, each column is a document, and each value is the count of that word in the document.
* Dimension of the matrix is number of words x number of documents. 
* We are trying to represent each word with m number of documents. The underlying conjecture is that words that are related often appears in similar documents.

#### 2.2 Window-Based Co-occurrence Matrix
* Each row is a word, each column is also a word. Each value is the count if this the column word is within window size position of the row word.
* Represent each word by the surrounding words.

#### Applying SVD on the Matrices
* Perform SVD on the matrix; Observe the singular values (the diagonal entries in the resulting S matrix), and cut them off at some index
k based on the desired percentage variance captured (could use this as dimensionality reduction method).

#### A couple problems for this approach
* The dimensions of the matrix change very often (new words are added very frequently and corpus changes in size).
* The matrix is extremely sparse since most words do not co-occur.
* The matrix is very high dimensional in general (≈ 10^6 × 10^6).
* Quadratic cost to train (i.e. to perform SVD).
* Requires the incorporation of some hacks on X to account for the drastic imbalance in word frequency.

### 3. Iteration Based Methods - Word2vec
The idea is to design a model whose parameters are the word vectors. Then, train the model on a certain objective. At every iteration
we run our model, evaluate the errors, and follow an update rule that has some notion of penalizing the model parameters that caused
the error. Thus, we learn our word vectors. 

#### Google word2vec model:
- 2 algorithms: continuous bag-of-words (CBOW) and skip-gram. CBOW aims to predict a center word from the surrounding context in
terms of word vectors. Skip-gram does the opposite, and predicts the distribution (probability) of context words from a center word.
- 2 training methods: negative sampling and hierarchical softmax. Negative sampling defines an objective by sampling negative examples, 
  while hierarchical softmax defines an objective using an efficient tree structure to compute probabilities for all the vocabulary.
  
##### Continuous Bag of Words Model (CBOW)
* Predicting a center word from the surrounding context, so output is a probability vector of words, and the y-true is a one hot encoder.
* Learn two vectors:
    * - v: (input vector) when the word is in the context; the i-th column of V is the n-dimensional embedded vector for word wi when it is an input to this model.
    * - u: (output vector) when the word is in the center; the j-th row of U is an n-dimensional embedded vector for word wj when it is an output of the model. 
* Minimize the cross-entropy loss = -ylog(y_hat); See source for detailed cost function
* Score z = Uv
* Update using stochastic gradient descent.

##### Skip-Gram Model
* Predicting the context based on the center word, so output is the probabilities (matrix) of observing each context word.
* Learn two vectors:
    * - v: (input vector) when the word is in the context; the i-th column of V is the n-dimensional embedded vector for word wi when it is an input to this model.
    * - u: (output vector) when the word is in the center; the j-th row of U is an n-dimensional embedded vector for word wj when it is an output of the model. 
* Minimize the cross-entropy loss = -ylog(y_hat); See source for detailed cost function
* Score z = Uv
* Key difference in formulating the objective function is that we invoke a Naive(Strong) Bayes assumption to break out the probabilities.
In other words, given the center word, all output words are completely independent.
* Update using stochastic gradient descent.

##### Negative Sampling
* For every training step, instead of looping over the entire vocabulary, we can just sample several negative examples! We "sample" from
a noise distribution (Pn(w)) whose probabilities match the ordering of the frequency of the vocabulary. To augment our formulation of
the problem to incorporate Negative Sampling, all we need to do is update the:
    * Objective function: 
    * Gradients
    * Update rules
* Particularly: We build a new objective function that tries to maximize the probability of a word and context being in the corpus data if it indeed is, 
  and maximize the probability of a word and context not being in the corpus data if it indeed is not.
  
([source](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf))


##### Hierarchical Softmax
* In practice, hierarchical softmax tends to be better for infrequent words, while negative sampling works better for frequent words and lower dimensional vectors.
* Hierarchical softmax uses a binary tree to represent all words in the vocabulary. Each leaf of the tree is a word, and there is a unique
path from root to leaf. In this model, there is no output representation for words. Instead, each node of the graph (except the root and the
leaves) is associated to a vector that the model is going to learn.
  
## Topic Modeling, Latent Dirichlet allocation
This is essentially a clustering problem - can think of both words and documents as being clustered.
LDA is a three-level hierarchical Bayesian model, in which each item of a collection is modeled as a finite mixture over an underlying set of topics. 
Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities.

Latent Dirichlet allocation (LDA) is a generative probabilistic model of a corpus. The basic idea is that documents are represented as random mixtures over latent topics, 
where each topic is characterized by a distribution over words.

LDA assumes the following generative process for each document w in a corpus D:
1. Choose N ∼ Poisson(ξ).
2. Choose θ ∼ Dir(α).
3. For each of the N words w_n:
(a) Choose a topic z_n ∼ Multinomial(θ).
(b) Choose a word wn from p(w_n |z_n,β), a multinomial probability conditioned on the topic z_n.
   
* α, β are parameters
* θ, z are unobserved/hidden variable
* Uses EM algorithm: E to compute the posterior probability; M to estimate parameters α, β

(Note in Naive Bayes: we have one topic per document)

([source](https://www.cl.cam.ac.uk/teaching/1213/L101/clark_lectures/lect7.pdf))
([source](https://www.seas.upenn.edu/~cis520/lectures/25a_LDA.pdf))

## RNN
### Deep Bidirectional RNNs
It is possible to make predictions based on future words by having the RNN model read through the corpus backwards.

### Gated Recurrent Unit
* Gated recurrent units are designed in a manner to have more persistent memory thereby making it easier for RNNs to capture long-term dependencies.
* Operational stages:
    * New memory generation
    * Reset Gate
    * Update Gate
    * Hidden state
    
### LSTM
* A bit different from Gated Recurrent Unit
* Can be used to encode and decode the sentence (seq2seq translation)
* Operational stages:
    * New memory generation
    * Input Gate
    * Forget Gate
    * Final memory generation
    * Output/Exposure Gate

([source](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf))


#### Attention Mechanism (Coupled with RNN)
* Key idea: different words have different importance, some words are more important than others
* Attention mechanisms make use of this observation by providing the decoder network with a look at the entire input sequence at every
decoding step; the decoder can then decide what input words are important at any point in time.
* The attention-based model learns to assign significance to different parts of the input for each step of the output. In the context of
translation, attention can be thought of as "alignment." Bahdanau et al. argue that the attention scores αij at decoding step i signify the
words in the source sentence that align with word i in the target. Noting this, we can use attention scores to build an alignment table –
a table mapping words in the source to corresponding words in the target sentence – based on the learned encoder and decoder from our
Seq2Seq NMT system.
  
* Encoder:
    *   Let (h1, . . . , hn) be the hidden vectors representing the input sentence. These vectors are the output of a bi-LSTM for instance, and
        capture contextual representation of each word in the sentence.
        
* Decoder:
    * We want to compute the hidden states si of the decoder using a recursive formula from previous hidden state, word generated from the previous step, 
      and c is a context vector that capture the context from the original sentence that is relevant to the time step i of the decoder.
    * The context vector ci captures relevant information for the i-th decoding time step (unlike the standard Seq2Seq in which there’s
    only one context vector). For each hidden vector from the original sentence hj, compute a score

### Attention based Only: transformer
* A model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.
The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
* Using stacked self-attention(multi-head attention) and point-wise, fully connected layers for both the encoder and decoder
* Can be trained using masking
* BERT: Pre-training uses a cloze task formulation where 15% of words are masked out and predicted
  
### Other training tips
* The train, tune, dev, and test sets need to be completely distinct (tuning set for setting hyperparameters)
* Learning rates:
    * very high rate: may explode the loss, and got the wrong direction of updating the weights
    * high rate: may get stuck in local optima
    * very low learning rate: learns slowly

([source](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf))
([source](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture10-QA.pdf))



