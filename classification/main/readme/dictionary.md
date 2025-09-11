- epoch = how many times the network will see each training example in Gradient descend algorithm
- batch/minibatch = Each iteration's sample of training data in a gradient descend algorithm

- Signal leakage/data leakage?
occurs when information from outside the training dataset is used to create the model in a way that gives it an unfair or unrealistic advantage. This typically happens when the model has access to data it wouldn’t realistically have during inference or deployment.
for example : songs from the same artist appears in both the training and test sets, solution :keep all data from the same group (e.g., artist) together in either training or test.

- Cross-entropy is a sort of measure for the distance from one probability distribution to another.

- Neural Network: a beautiful biologically-inspired programming paradigm which enables a computer to learn from observational data.

- Deep Learning: a powerful set of techniques for learning in neural networks.

- feedforward neuralnetwork: when the output from one layer is used as inputs for the next layer.
- recurrent neural network: feedback loops are possible, the idea is to have some neurons that fire for a limited time before becoming inactive.
- heuristic: it's an approximation used to make problem-solving more efficient when classic methods are too slow or impractical.
- loss function == objective function == cost function
- zip() function: The zip() function combines two (or more) lists element-by-element.