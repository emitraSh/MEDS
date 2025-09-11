# <u>Classification Models:</u>

### Naive Bayes Classification:
<u>Model:</u>
- It uses a generative approach.
- It simply assumes that the features are conditionally independent.(covariance matrix is diagonal matrix)
therefor for each class k we have:
  - p(x | C<sub>k</sub>) = ∏ p(x<sub>j</sub> | C<sub>k</sub>)
- The form of the class-conditional density depends on the type of each feature
    - Real value --> Gaussian Distribution
    - Binary features : Bernoulli Distribution
    - Categorical feature : Multinomial Distribution
- To compute priors we can use MLE or Bayesian approaches
    - MLE :  π<sub>c</sub> = N<sub>c</sub> / N
    - Bayes approach : depending on the data-type we can use different distributions.

<u>Prediction:</u>
- p(C<sub>k</sub> | x, D) ∝ p(C<sub>k</sub> | D)∏ p(x<sub>(j)</sub>| C<sub>k</sub> , D)



### Gaussian Discriminative Analysis:
- p(x | y = c , θ) = N (µ<sub>c</sub> , Σ<sub>c</sub> )
  - Σ<sub>c</sub> / covariance matrix : Quadratic matrix row =column = number of features
  - µ<sub>c</sub> : an array of means for each feature existing in the dataset
  - N (µ<sub>c</sub> , Σ<sub>c</sub> ):= ((2π)<sup>D/2</sup>|Σ|<sup>1/2</sup>)<sup>-1</sup>exp[−((x − µ)<sup>T</sup> Σ<sup>-1</sup>(x − µ))/2]
- <u> Quadratic discriminant analysis:</u> each class has it own covariance matrix
- <u> Linear discriminant analysis: </u>the covariance matrix is forced to be the same for all classes
- In this case the features are not assumed to be independent, their dependence is portrayed in the covariance matrix 
- It is a generative model even though called discriminative


### Decision Tree Classification:
It's a binary tree where each node represents a yes/no question based on features. At each step, the data is split until we reach pure groups that mostly belong to the same class. The tree can then classify new data points—essentially, it's just nested if-else statements!
- at each decision node the condition is determined by a variable called Information-Gain which is determined by Gini Index or Entropy or a combination of those two
    - Entropy :   Σ -p<sub>i</sub> *  log(p<sub>i</sub>)
    - Gini Index : 1-Σ(p<sub>i</sub>)<sup>2</sup>     
    - (i represents each unique data in an array of values)
    - if we
    - Information Gain = result of Gini/Entropy of parents minus sum of the weighted Gini/Entropy results for left and right child.
    - For decision tree model it is important to set a maximum depth to avoid getting into loops
    - also it is important to set a minimum sample split to avoid small amount of data influencing the classification

### Logistic Regression:
Logistic regression is a method used to predict yes/no answers (also called binary classification).
- It uses sigmoid function for its classification.
- it calculates the numbers between 0 and 1, the closer to 1, the probability of success is higher.
- it has a threshold at 0.5.
- Forward pass = the equation that sigmoid function is applied to -> x.w + b
- It uses optimization algorithms to train the model such as:
  - Steepest Decent
  - Newton's Method
  - Iterative reweighed least square
- parameters: 
  - learning rate (float)
  - number of iteration 

# Why Use Standardization in Machine Learning?
- Mean Centering -> it helps with capturing <u>relative variation</u> in data
- Scale Invariance -> it avoids influence of large scale features by dividing each feature by its standard deviation
- Improves convergence in <u>gradiant based algorithms </u>
- comparability
- Regularization

- - - - - How to Standardize? 
calculate the mean and standard deviation of each feature and apply (x-mean)/sd to each x.
