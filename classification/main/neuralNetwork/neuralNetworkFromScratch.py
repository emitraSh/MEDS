import numpy as np
import random

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1],sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] #returns arrays that follow StandardNormal distribution
        #weights and biases are set randomly at first
        #np.random.randn gives random numbers drawn from the standard normal distribution which is a good start for stochastic gradient descend.



    def sigmoid_prime(self,z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def feedforward(self, a):
        """ first "a" is the input array in the shape of (n,1). n is indicating inputs to the network """
        for w,b in zip(self.weights,self.biases):
            a = self.sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,
            df_train,
            epochs,
            mini_batch_size,
            learning_rate,
            df_test=None):

        """
                Train the neural network using mini-batch stochastic
                gradient descent.  The "training_data" is a list of tuples
                "(x, y)" representing the training inputs and the desired
                outputs.  The other non-optional parameters are
                self-explanatory.  If "test_data" is provided then the
                network will be evaluated against the test data after each
                epoch, and partial progress printed out.  This is useful for
                tracking progress, but slows things down substantially.
        """
        
        n = len(df_train)

        for epoch in range(epochs):
            random.shuffle(df_train) #shuffling happens in_place so the original order doesn't change
            mini_batches = [ df_train[k: k +mini_batch_size] for k in range(0, n, mini_batch_size) ]

            for batch in mini_batches:
                self.update_mini_batch(batch, learning_rate) # single step of gradient descend

            """
                If the optional argument test_data is supplied, 
                then the program will evaluate the network after each epoch of training,
                and print out partial progress.
                This is useful for tracking progress, 
                but slows things down substantially.      
            """

            if df_test:
                n_test = len(df_test)
                print
                "Epoch {0}: {1} / {2}".format(
                    epoch, self.evaluate(df_test), n_test)
            else:
                print
                "Epoch {0} complete".format(epoch)

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        the end result of this function is updated weights and biases after each epoch
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases] # : gradient / upward triangle
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            print(f"what is the shape of - {x.shape} - {y.shape}________ ")
            delta_nabla_b, delta_nabla_w = self.backprop(y, x)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            #these parts are equivalent for : Δv=−η∇C -> v′=v−(learning rate)*∇C :
            self.weights= [w - learning_rate* nw for w, nw in zip(self.weights, nabla_w)]
            self.biases= [b - learning_rate* nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, y, x):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
                gradient for the cost function C_x.  ``nabla_b`` and
                ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
                to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)









