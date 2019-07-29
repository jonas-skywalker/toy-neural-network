import neuralnetwork
import random
import my_matrix_lib as matrix

outputs = []


def barrier():
    print("=====================================")


data_set = [[[0.01, 0.01], [0.01]],
            [[0.99, 0.99], [0.01]],
            [[0.99, 0.01], [0.99]],
            [[0.01, 0.99], [0.99]]]


nn = neuralnetwork.NeuralNetwork([2, 2, 2, 1])

for mat in nn.weights:
    matrix.Matrix.print_mat(mat)
    barrier()

for mat in nn.biases:
    matrix.Matrix.print_mat(mat)
    barrier()

# matrix.Matrix.print_mat(nn.feed_forward([0.99]))
# nn.linear_regression_gradient_descent([0.5], [0.9])

for i in range(50000):
    data = random.choice(data_set)
    input_list = data[0]
    target_list = data[1]
    nn.linear_regression_gradient_descent(input_list, target_list)

print("////////////////////////////////")

outputs.append(nn.feed_forward([0.01, 0.01]))
outputs.append(nn.feed_forward([0.99, 0.99]))
outputs.append(nn.feed_forward([0.99, 0.01]))
outputs.append(nn.feed_forward([0.01, 0.99]))

for mat in outputs:
    matrix.Matrix.print_mat(mat)
