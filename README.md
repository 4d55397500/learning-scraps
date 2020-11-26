learning_scraps
---

various learning scraps

* [**multi-instance learning**](https://github.com/redwrasse/multi-instance-learning): Implements [From Group to Individual Labels using Deep Features](https://arxiv.org/pdf/1411.3128.pdf). idea: transfer knowledge of labels on groups (subsets of data) to learned labels on individual points. often labels on groups are present/cheap, labels on individuals are expensive/rare.

Example: Suppose known voting results by neighborhood, as well as features of (but not specific voting result) of individuals. Multi-instance learning uses this information to infer voting likelihood of each individual.
* [**heteroscedastic model**](https://gist.github.com/redwrasse/1281b12a7012ad9e699842f2701eb8a9): parameterized variance in discriminative gaussian
* [**gradient desecent gaussian**](https://gist.github.com/redwrasse/310189d41dc3bab76ac5956e654286a8): Classic generative model, gradient descent on mean, standard deviation
* [**gradient descent on gaussian mixture model**](https://gist.github.com/redwrasse/e46976d3fc2df7528742b6f55a79b315): Demonstrated poor training on direct gradient descent on full generative model for a GMM, motivating the EM algorithm. In this case, exponential terms summed inside the logarithm cause the poor training.
* [**autoencoders/semantic hashing**](https://github.com/redwrasse/autoencoders): Various autoencoders, semantic hashing.
* [**autoregressive convolutional layer**](https://gist.github.com/redwrasse/9e91904fcd63511a1350af374b644396): AR(2) model trained with convolutional layer 
* [**backprop-example**](https://gist.github.com/redwrasse/54ce1fb6731b9bd688647f5b5e1f5dfc): Backprop algorithm on example functional.
* [**array-reinforcement-learning**](https://gist.github.com/redwrasse/dd5dd4924129d338b3a5ab6f6ac74d1b): Value iteration algorithm on a 1d arrray.
* [**minhash**](minhash/minhash.py): verifies the relationship between minhash distance and Jaccard similarity
* [**singular value decomposition**](svdtext/svdtext.py): svd for similarity between words in sentences
* [**streaming posterior update**](./posterior/cointoss.py): iterative posterior update
* [**kernel perceptron**](./kernel_perceptron/kernel_perceptron.py): kernel perceptron in Tensorflow
* [**dual perceptron**](./dual_perceptron/dual_perceptron.py): dual perceptron in Tensorflow
* [**logistic regression**](./log_reg/log_reg.py): logistic regression in Tensorflow
* [**local minimum**](./local_min/local_min.py): Find a local minimum
* [**local minimum with norm constraint**](./local_min_constraint/local_min_constraint.py): Find a local minimum  with a constraint
