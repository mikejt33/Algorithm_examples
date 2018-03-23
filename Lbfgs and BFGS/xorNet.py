#!/usr/bin/python
import numpy as np
from chapter_3_algorithms import line_search, zoom, interpolate
from bfgs import update_inverse_hessian, algorithm7_4, add_to_buffer
import signal
import sys

'''
This script implements a fully connected neural net with 1 hidden
layer, and 1 output layer of one node, to train an XOR gate. Uses
gradient descent on the MSE cost function.
'''

# Set to True for print debugging
PRINT_DEBUG_MESSAGES = True

# Training meta parameters.
GRADIENT_STOP_EPSILON = 1e-8 # stop when norm(grad) < eps
LINE_SEARCH_C1 = 1e-4 #
LINE_SEARCH_C2 = 0.9 # Recommended value for quasi-newton

def debug(s):
    if PRINT_DEBUG_MESSAGES is True:
        print(s)

'''
Neuron class, which represents a summation and sigmoid node. Keeps track of its
local weights, as well as which other Neurons feed into it. All weights are initially randomized to a value between -0.5 and 0.5.

Makes heavy use of recursion for that:

ＥＬＥＧＡＮＣＥ
Ｌ
Ｅ
Ｇ
Ａ
Ｎ
Ｃ
Ｅ
'''
class Neuron():

    def __init__(self):

        # We start off with just 1 weight, the bias of this Neuron.
        self.weights = []
        self.addWeights(1)

        # This vector will store the weights for this node.
        # List of neurons which feed into this neuron
        self.childNeurons = []


    ''' Add a neuron which feeds into this neuron.  '''
    def addChild(self, childNeuron):

        # Add the neuron to our list
        self.childNeurons.append(childNeuron)

        self.addWeights(1)


    '''
    Add num weights to the weights vector. Useful as an internal helper
    function, but also allows the Net class to tell the very first layer of
    Neurons about weights to have ready for input variables.
    '''
    def addWeights(self, num):

        # Initialize weights randomly
        new_weights = np.random.random(size = num) - 0.5

        # Append new weights to end of weights vector
        self.weights = np.concatenate((self.weights, new_weights))

        try:
            self.numWeights += num
        except AttributeError:
            self.numWeights = num

    '''
    Return how many weights (connections) are in the subtree starting at
    this Neuron. Recursive!
    '''
    def getNumWeights(self):
        ret = len(self.weights)
        for childNeuron in self.childNeurons:
            ret += childNeuron.getNumWeights()
        return ret

    '''
    Return the weights vector for this subtree
    Recursion! Elegance!
    '''
    def getWeights(self):
        w = self.weights

        # Loop through the childNeurons from bottom to top
        for childNeuron in reversed(self.childNeurons):

            # Recursively grab this Neuron's weights
            subWeights = childNeuron.getWeights()

            # Stack those weights on top of the current weight
            w = np.concatenate((subWeights,w), axis=0)

        # Now we've built the full weights vector for every Neuron below this one
        return w


    ''' Add the vector update to the weights vector. Recursive! '''
    def updateWeights(self, update):

        # Snip off the lowest elements of the vector to change this Neuron's
        # weights with
        update, local_update = np.split(update, [-1 * self.numWeights], axis=0)

        # Apply the local update
        self.weights = self.weights + local_update # TODO: ensure dimensions match (reshape)?

        numChildren = len(self.childNeurons)
        if numChildren == 0:
            return
        # implicit else

        # Split the update into chunks for each of the child neurons
        # If the update vector is of the right size, then these chunks
        # should all be the same size
        subUpdates = np.split(update, numChildren, axis=0)

        # Loop through the childNeurons from bottom to top
        for childNeuron, childUpdate in zip(reversed(self.childNeurons),
                                            reversed(subUpdates)):

            # Update the child's weights
            childNeuron.updateWeights(childUpdate)

        return

    '''
    Do the same thing as updateWeights, but just replace the weights with
    this new vector.
    '''
    def setWeights(self, new_weights):

        # Snip off the lowest elements of the vector to change this Neuron's
        # weights with
        new_weights, local_weights = np.split(new_weights, [-1 * self.numWeights], axis=0)

        # Set the local weights
        self.weights = local_weights # TODO: ensure dimensions match (reshape)?

        numChildren = len(self.childNeurons)
        if numChildren == 0:
            return
        # implicit else

        # Split the update into chunks for each of the child neurons
        # If the update vector is of the right size, then these chunks
        # should all be the same size
        subUpdates = np.split(new_weights, numChildren, axis=0)

        # Loop through the childNeurons from bottom to top
        for childNeuron, childUpdate in zip(reversed(self.childNeurons),
                                            reversed(subUpdates)):

            # Update the child's weights
            childNeuron.setWeights(childUpdate)

        return


    '''
    Given neural net input, process the output of this node. First recursively
    calculates the output for all child Neurons, which becomes the input to
    this node. If this node has no children, then the neural net input is used.
    The input is then weighted adn summed, and passed through a sigmoid
    function. This scalar output is returned.

    net_input: A numpy ndarray with the inputs to the whole Neural net. Can
                be a column or row vector, as it will be reshaped.
    '''
    def process(self, net_input):

        if len(self.childNeurons) == 0:
            input_v = net_input
        else:
            input_v = []
            for childNeuron in self.childNeurons:
                input_v.append(childNeuron.process(net_input))
            input_v = np.asarray(input_v)

        # Reshape the input. Should be 1 less than the length of the weights
        # matrix (no input for the bias)
        input_v = input_v.reshape(len(self.weights) - 1)

        # Insert a 1 at the top of the vector, for the bias
        input_v = np.insert(input_v, 0, 1)

        # Store this input vector for future use in the gradient computation.
        self.last_input_v = input_v

        # Augmented input vector, dotted with weight vector, gives this
        # neuron's weighted sum of the inputs. This scalar is then passed
        # through the sigmoid function, and that is the final scalar output,
        # between 0 and 1. We store that output for future use in the gradient
        # computation.
        s = np.dot(input_v, self.weights)
        self.last_output = sigmoid(s)

        return self.last_output



    '''
    ASSUMES A FORWARD PASS HAS BEEN DONE BEFORE CALLING THIS.
    Returns a local gradient. Recursively calls child nodes and uses their
    sub-gradients to construct this sub-gradient. If this node is the root
    node, then the return will be the full gradient of the output node of the
    net.
    '''
    def getGradient(self):

        '''
        First construct the local gradient, for the derivative of this node
        with respect to its weights. last_output is a scalar, being multiplied component-wise into the
        last_input_v vector. This formula comes from the chain rule, and the
        definition of the derivative of the sigmoid function.
        '''
        local_grad = self.last_output * (1 - self.last_output)
        g = local_grad * self.last_input_v

        # Loop through the childNeurons from bottom to top
        for i in range(len(self.childNeurons) - 1, -1, -1):

            childNeuron = self.childNeurons[i]
            childNeuron_weight = self.weights[i+1]

            # Recursively grab this Neuron's gradient
            subGrad = childNeuron.getGradient()

            # Scale the entire gradient by the local_grad, and its weight.
            # This comes from the chain rule.
            subGrad = local_grad * childNeuron_weight * subGrad

            # Stack that subgradient on top of the current gradient
            g = np.concatenate((subGrad,g), axis=0)

        # Now we've built the gradient for this Neuron
        return g



'''
Class for storing our graph of the neural net architecture.
Fully connected
Same number and type of nodes within each hidden layer
each neuron is a sum (with bias) and sigmoid
all inputs are passed to every neuron in the first hidden
layer.
only one output node which gives a vector output
'''
class Net():

    '''

    inputs: matrix where each row is an observation
    outputs: vector of expected output for each observation
    num_neurons_per_layer: how many neurons per hidden layer
    num_hidden_layers: how many processing layers
    '''
    def __init__(self, inputs, outputs, num_neurons_per_layer,
                    num_hidden_layers):
        self.inputs = inputs
        self.outputs = outputs
        self.num_neurons_per_layer = num_neurons_per_layer
        self.num_hidden_layers = num_hidden_layers

        # Create our graph of Neurons
        prior_layer = []
        for i in range(num_hidden_layers):
            this_layer = []
            for j in range(num_neurons_per_layer):
                new_neuron = Neuron()
                this_layer.append(new_neuron)

                if len(prior_layer) == 0:
                    # This is the first hidden layer, so tell these neurons
                    # about the weights they should have ready for the input
                    new_neuron.addWeights(self.inputs.shape[1])
                else:
                    for prior_neuron in prior_layer:
                        new_neuron.addChild(prior_neuron)

            prior_layer = this_layer

        # We assume this particular net will only have a single node in the final
        # output layer
        self.rootNeuron = Neuron()
        for prior_neuron in prior_layer:
            self.rootNeuron.addChild(prior_neuron)

        # Store initial weights to see how things change
        self.init_weights = self.getWeights()

        # Catch Ctrl-C interrupts, and print output before shutting down.
        # Useful if the code is hanging due to a bug
        def signal_handler(signal, frame):
            print('Caught an interrupt!')
            self.show_output()
            sys.exit(0)

        # Attach the signal handler
        signal.signal(signal.SIGINT, signal_handler)

    ''' Add the update vector to the net's weights. '''
    def updateWeights(self, update):

        # This call will recurse down the whole graph
        self.rootNeuron.updateWeights(update)

    '''
    input_v is an array with the x_i's.
    '''
    def feedForward(self, input_v):
        return self.rootNeuron.process(np.asarray(input_v))

    ''' Returns the weights of the whole net as a p by 1 vector. '''
    def getWeights(self):
        return self.rootNeuron.getWeights()

    ''' Set the weights of the net. '''
    def setWeights(self, new_weights):
        self.rootNeuron.setWeights(new_weights)
        self.init_weights = new_weights # store weights

    ''' Return how many weights exist in this net. '''
    def getNumWeights(self):
        return self.rootNeuron.getNumWeights()

    '''
    Return the evaluation of the MSE cost function at the net's
    current weights.
    '''
    def getCost(self):

        cost = 0

        # Sum the squared error over all observations
        # self.outputs is assumed to be of shape (n,)
        for observation, yi in zip(self.inputs, self.outputs):

            # Feed forward gives the output of the net for a particular
            # input, with the weights the net currently has
            cost += (yi - self.feedForward(observation)) ** 2

        return 0.5 * cost


    '''
    Return the gradient of the MSE cost function at the net's
    current weights.
    '''
    def getCostGradient(self):

        cost_grad = None

        # Sum the squared error over all observations
        # self.outputs is assumed to be of shape (n,)
        for observation, yi in zip(self.inputs, self.outputs):

            # Calling feed forward allows the nodes to store their outputs for
            # the getGradient call
            net_output = self.feedForward(observation)

            error = yi - net_output

            if cost_grad is None:
                cost_grad = (-1 * error * self.rootNeuron.getGradient())
            else:
                cost_grad += (-1 * error * self.rootNeuron.getGradient())

        return cost_grad


    ''' 
    Print out how well the net fits the input and expected output. Maybe
    don't do this if there are a large number of data points. 
    '''
    def show_output(self):

        # Show initial and final parameter vectors in weight-space
        print("Initial weights: ", self.init_weights)
        print("Weights after training: ", self.getWeights())

        # For each observation, show error
        for xi, yi in zip(self.inputs,self.outputs):
            out = self.feedForward(xi)
            print("For observation %s, Expected: %f, Output: %f" % (xi, yi, out))

        print("Final cost: %.15f" % self.getCost())


    '''
    Train our neural network, using gradient descent with a line
    search.
    '''
    def trainGradDescend(self):

        # Calculate initial cost
        cost = self.getCost()

        # Set initial gradient and gradient norm
        grad_k = self.getCostGradient()
        grad_norm = np.linalg.norm(grad_k)

        # Break once the gradient's norm is sufficiently close to 0
        while grad_norm > GRADIENT_STOP_EPSILON:

            # Print debug info
            debug("COST = %.8f" % cost)
            debug("GRADIENT NORM = %.8f" % grad_norm)

            # Set descent direction to be negative of gradient
            p_k = - grad_k

            # The phi function is a cross-section of our cost function from our
            # current point in weight-space in the direction of p_k
            def phi(alpha):

                # Add alpha * p_k to the weights
                temp_update = alpha * p_k
                self.updateWeights(temp_update)

                cost_at_alpha = self.getCost()

                # Reset the weights
                self.updateWeights(-1 * temp_update)

                return cost_at_alpha

            # The derivative of phi w.r.t. alpha is the gradient of the cost
            # function evaluated at our current point in weight-space plus
            # alpha * p_k, dotted with p_k (by the chain rule).
            def phi_prime(alpha):

                # Add alpha * p_k to the weights
                temp_update = alpha * p_k
                self.updateWeights(temp_update)

                # Calculate the gradient of the net with these new weights
                cost_grad_at_alpha = self.getCostGradient()

                # Reset the weights
                self.updateWeights(-1 * temp_update)

                ret = np.dot(cost_grad_at_alpha, p_k) # TODO: check shapes

                return ret

            # Call the line_search method which uses Algorithm 3.5 from the
            # book, and finds a point along phi which satisfies the Wolfe
            # conditions.
            alpha = line_search(phi, phi_prime, LINE_SEARCH_C1, LINE_SEARCH_C2)
            debug("ALPHA = %F" % alpha)

            # Take a step along p_k to descend the cost function
            self.updateWeights(alpha * p_k)

            # Find new gradient
            grad_k = self.getCostGradient()

            # Compute new cost and gradient norm
            cost = self.getCost()
            grad_norm = np.linalg.norm(grad_k)

            # repeat


    '''
    Train our neural network, using BFGS with a line
    search.
    '''
    def trainBFGS(self):

        # Calculate initial cost
        cost = self.getCost()

        # Set initial gradient and gradient norm
        grad_k = self.getCostGradient()
        grad_norm = np.linalg.norm(grad_k)

        # Set initial H_k
        n = self.getNumWeights() # how many parameters in this net
        H_k = np.identity(n)

        # Break once the gradient's norm is sufficiently close to 0
        while grad_norm > GRADIENT_STOP_EPSILON:

            # Print debug info
            debug("COST = %.8f" % cost)
            debug("GRADIENT NORM = %.8f" % grad_norm)

            # Calculate descent direction using minimizer of quadratic
            # approximation to our cost function (quasi-Newton method)
            p_k = - H_k.dot(grad_k)

            # The phi function is a cross-section of our cost function from our
            # current point in weight-space in the direction of p_k
            def phi(alpha):

                # Add alpha * p_k to the weights
                temp_update = alpha * p_k
                self.updateWeights(temp_update)

                cost_at_alpha = self.getCost()

                # Reset the weights
                self.updateWeights(-1 * temp_update)

                return cost_at_alpha

            # The derivative of phi w.r.t. alpha is the gradient of the cost
            # function evaluated at our current point in weight-space plus
            # alpha * p_k, dotted with p_k (by the chain rule).
            def phi_prime(alpha):

                # Add alpha * p_k to the weights
                temp_update = alpha * p_k
                self.updateWeights(temp_update)

                # Calculate the gradient of the net with these new weights
                cost_grad_at_alpha = self.getCostGradient()

                # Reset the weights
                self.updateWeights(-1 * temp_update)

                ret = np.dot(cost_grad_at_alpha, p_k) # TODO: check shapes

                return ret

            # Call the line_search method which uses Algorithm 3.5 from the
            # book, and finds a point along phi which satisfies the Wolfe
            # conditions.
            alpha = line_search(phi, phi_prime, LINE_SEARCH_C1, LINE_SEARCH_C2)
            debug("ALPHA = %F" % alpha)

            # Take a step along p_k to descend the cost function
            s_k = alpha * p_k
            self.updateWeights(s_k)

            # Find new gradient, and y_k, the difference between gradients
            grad_kp1 = self.getCostGradient()
            y_k = grad_kp1 - grad_k
            grad_k = grad_kp1

            # Compute H_{k+1}
            H_k = update_inverse_hessian(H_k, s_k, y_k)

            # Compute new cost and gradient norm
            cost = self.getCost()
            grad_norm = np.linalg.norm(grad_k)

            # repeat


    def trainLBFGS(self):

        # Calculate initial cost
        cost = self.getCost()

        # Set initial gradient and gradient norm
        grad_k = self.getCostGradient()
        grad_norm = np.linalg.norm(grad_k)

        # Store identity matrix as I
        n = self.getNumWeights() # how many parameters in this net
        I = np.identity(n)

        # Declare first hk_0 as the identity
        hk_0 = I

        # Break once the gradient's norm is sufficiently close to 0
        while grad_norm > GRADIENT_STOP_EPSILON:

            # Print debug info
            debug("COST = %.8f" % cost)
            debug("GRADIENT NORM = %.8f" % grad_norm)


            # Calculate descent direction using rank-m approximation of
            # inverse Hessian
            p_k = - algorithm7_4(grad_k, hk_0)

            # The phi function is a cross-section of our cost function from our
            # current point in weight-space in the direction of p_k
            def phi(alpha):

                # Add alpha * p_k to the weights
                temp_update = alpha * p_k
                self.updateWeights(temp_update)

                cost_at_alpha = self.getCost()

                # Reset the weights
                self.updateWeights(-1 * temp_update)

                return cost_at_alpha

            # The derivative of phi w.r.t. alpha is the gradient of the cost
            # function evaluated at our current point in weight-space plus
            # alpha * p_k, dotted with p_k (by the chain rule).
            def phi_prime(alpha):

                # Add alpha * p_k to the weights
                temp_update = alpha * p_k
                self.updateWeights(temp_update)

                # Calculate the gradient of the net with these new weights
                cost_grad_at_alpha = self.getCostGradient()

                # Reset the weights
                self.updateWeights(-1 * temp_update)

                ret = np.dot(cost_grad_at_alpha, p_k) # TODO: check shapes

                return ret

            # Call the line_search method which uses Algorithm 3.5 from the
            # book, and finds a point along phi which satisfies the Wolfe
            # conditions.
            alpha = line_search(phi, phi_prime, LINE_SEARCH_C1, LINE_SEARCH_C2)
            debug("ALPHA = %F" % alpha)

            # Take a step along p_k to descend the cost function
            s_k = alpha * p_k
            self.updateWeights(s_k)

            # Find new gradient, and y_k, the difference between gradients
            grad_kp1 = self.getCostGradient()
            y_k = grad_kp1 - grad_k
            grad_k = grad_kp1

            # Save this pair into our circular buffer
            add_to_buffer(s_k, y_k)

            # Compute new cost and gradient norm
            cost = self.getCost()
            grad_norm = np.linalg.norm(grad_k)

            # gamma_k gives us a good sense of how to scale hk_0
            gamma_k = s_k.dot(y_k) / y_k.dot(y_k)

            # Use gamma_k to set a new hk_0
            hk_0 = gamma_k * I

            # repeat

'''
Sigmoid function, returns values between 0 and 1
Numpy allows us to vectorize this function, so we can return
a vector of sigmoids.
'''
def sigmoid(x):
    return 1.0 / (1 + np.exp(-1*x))


def search():

    #np.random.seed(42) # consistency of random init for debugging

    #np.random.seed(42) # doesn't work: 0, .5, 1, .5
    #np.random.seed(99) # does work
    #np.random.seed(79) # does work for lbfg
    #np.random.seed(13)# .5, .5, 1, 0
    #np.random.seed(22) # 0.3's??

    # Hardcode what we know about an XOR gate
    X = np.asarray([[0,0], [0,1], [1,0], [1,1]]) # predictors
    Y = np.asarray([0, 1, 1, 0]) # response

    # Create our neural nets, with 1 hidden layer and 2 neurons
    # per hidden layer. We'll train each net in a different way
    g1 = Net(X, Y, 2, 1)
    g2 = Net(X, Y, 2, 1)
    g3 = Net(X, Y, 2, 1)

    # Use the same starting weights as g1, for comparison
    w = g1.getWeights()
    g2.setWeights(w)
    g3.setWeights(w)

    # Train our nets using various methods
    # Show output after each training, in case one of them hangs.
    g1.trainGradDescend()
    print("GRADIENT DESCENT OUTPUT:")
    g1.show_output()

    g2.trainBFGS()
    print("BFGS OUTPUT:")
    g2.show_output()

    g3.trainLBFGS()
    print("LBFGS OUTPUT:")
    g3.show_output()

if __name__ == '__main__':
    search()

