"""
Neural Network class file to create a neural network with a list as the structure
Example:
    list = [100, 20, 30, 10]
    neural_net = NeuralNetwork(list)
This example creates a Neural Network with 100 input nodes, two hidden layers with 20 and 30 nodes
and finally 10 output nodes.
"""
import my_matrix_lib as matrix
import math
import random


# Activation Function
def sigmoid(x):
    return 1/(1 + math.e**(-x))


class NeuralNetwork:
    def __init__(self, structure, lr=0.1, af=sigmoid):
        self.outputs = []
        self.inputs = []
        self.errors = []
        self.weights = []
        self.biases = []
        self.lr = lr
        self.af = af
        # check if structure is ok
        if len(structure) < 2:
            print(">>> Cannot create Neural Network from given Structure!")
        elif len(structure) == 2:
            self.input_nodes = structure[0]
            self.output_nodes = structure[1]
            self.weights.append(matrix.Matrix(self.output_nodes, self.input_nodes, mat_type="random"))
            self.biases.append(matrix.Matrix(self.output_nodes, 1, mat_type="random"))

        else:
            # Declaration of input layer
            self.input_nodes = structure[0]
            self.first_hidden_nodes = structure[1]
            self.weights.append(matrix.Matrix(self.first_hidden_nodes, self.input_nodes, mat_type="random"))
            self.biases.append(matrix.Matrix(self.first_hidden_nodes, 1, mat_type="random"))

            # Declaration of hidden layers
            for i in range(1, len(structure) - 2):
                self.weights.append(matrix.Matrix(structure[i + 1], structure[i], mat_type="random"))
                self.biases.append(matrix.Matrix(structure[i + 1], 1, mat_type="random"))

            # Declaration of output layer
            self.output_nodes = structure[-1]
            self.last_hidden_nodes = structure[-2]
            self.weights.append(matrix.Matrix(self.output_nodes, self.last_hidden_nodes, mat_type="random"))
            self.biases.append(matrix.Matrix(self.output_nodes, 1, mat_type="random"))

    def feed_forward(self, list_input):

        # Convert inputs into input vector
        mat_input = matrix.Matrix.createVector(list_input)

        matrix.Matrix.print_mat(mat_input)

        input = mat_input
        self.outputs.append(input)

        # loop through the matrices and biases and feed the input through the neural network
        for i in range(len(self.weights)):
            print("loop " + str(i))
            product = matrix.Matrix.mat_mul(self.weights[i], input)
            new_input = matrix.Matrix.vector_add(product, self.biases[i])
            matrix.Matrix.print_mat(new_input)
            self.inputs.append(new_input)
            print("******************")
            output = matrix.Matrix.apply_func(new_input, self.af)
            matrix.Matrix.print_mat(output)
            print("******************")
            input = output
            self.outputs.append(output)
        # SM = matrix.Matrix.soft_max(output)
        return output  # SM

    def linear_regression_gradient_descent(self, list_input, target_input):
        self.outputs = []
        self.inputs = []
        self.errors = []
        # Convert target to vector object
        target = matrix.Matrix.createVector(target_input)
        print("Target:")
        matrix.Matrix.print_mat(target)
        # feed forward the input
        print("/////////////Feedworward////////////")
        output = self.feed_forward(list_input)
        print("/////////////Feedforward////////////")
        print("Output:")
        matrix.Matrix.print_mat(output)

        error = matrix.Matrix.vector_sub(output, target)
        print("Calculated output error:")
        matrix.Matrix.print_mat(error)

        self.errors.append(error)

        # calculate all errors
        for i in range(len(self.weights)-1, 0, -1):
            error = matrix.Matrix.mat_mul(matrix.Matrix.transpose(self.weights[i]), self.errors[0])
            self.errors.insert(0, error)

        print("All calculated errors:")
        for mat in self.errors:
            matrix.Matrix.print_mat(mat)

        print("All calculated outputs:")
        for mat in self.outputs:
            matrix.Matrix.print_mat(mat)

        print("All weights:")
        for mat in self.weights:
            matrix.Matrix.print_mat(mat)
        print("All biases:")
        for mat in self.biases:
            matrix.Matrix.print_mat(mat)

        # backpropagate and start with the adjustment of the last weight matrix
        for i in range(len(self.weights) - 1, -1, -1):
            print("Backpropagation " + str(i))

            # updating weights
            # calculating the derivative of the activation function
            first = matrix.Matrix.mult(self.outputs[i + 1], -1)

            second = matrix.Matrix.add(first, 1)

            derivative = matrix.Matrix.vector_mult(self.outputs[i + 1], second)
            print("derivative:")
            matrix.Matrix.print_mat(derivative)

            gradient = matrix.Matrix.vector_mult(self.errors[i], derivative)
            print("third:")
            matrix.Matrix.print_mat(gradient)

            fourth = matrix.Matrix.mat_mul(gradient, matrix.Matrix.transpose(self.outputs[i]))
            print("fourth:")
            matrix.Matrix.print_mat(fourth)

            delta_weights = matrix.Matrix.mult(fourth, self.lr)
            print("delta weights:")
            matrix.Matrix.print_mat(delta_weights)

            self.weights[i] = matrix.Matrix.mat_add(self.weights[i], delta_weights)
            # updating biases
            delta_biases = matrix.Matrix.mult(gradient, self.lr)
            print("delta biases:")
            matrix.Matrix.print_mat(delta_biases)
            self.biases[i] = matrix.Matrix.vector_add(self.biases[i], delta_biases)
        print("All weights after:")
        for mat in self.weights:
            matrix.Matrix.print_mat(mat)
        print("All biases after:")
        for mat in self.biases:
            matrix.Matrix.print_mat(mat)
