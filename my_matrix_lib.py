"""
My Library for linear Algebra/ Matrix math for machine learning purposes
Create Matrices either with data or with number of columns, rows and matrix initialisation type
Example 1:
    data1 = [[3, 5, 1, 6],
            [2, 3, 6, 7],
            [5, 2, 5, 3],
            [2, 8, 7, 7]]
    A = Matrix(data=data1)
This returns a Matrix object with the dimensions and values as given in the data:

Example 2:
    A = Matrix(3, 6, mat_type="random")
This returns a Matrix object with 3 rows and 6 columns and random values between -1 an 1.

Matrix types are:
    "random": random values between -1 and 1
    "zeros": all zero matrix
    "gauss_random": random normal distribution values with mu=0 and sigma=columns ** (-1/2)

Things that should be added in the future for neuro-evolution maybe in a subclass called neuro:
    def crossover
    def mutation
    def inverse
"""
import random
import math


class Matrix:
    # initialize the Matrix object either with data or random with given rows and columns
    def __init__(self, rows=0, cols=0, mat_type="random", data=None):
        if data:
            # if isinstance(data, list):
            #    self.rows = len(data)
            #    self.cols = 1
            #    self.zeros()
            #    for i in range(len(data)):
            #        self.matrix_data[i][0] += data[i]
            self.matrix_data = data
            self.rows = len(data)
            self.cols = len(data[0])
        else:
            self.rows = rows
            self.cols = cols
            self.mat_type(mat_type)

    # show the Matrix object
    @staticmethod
    def print_mat(A):
        if A:
            if type(A) is Matrix:
                for i in range(A.rows):
                    print(A.matrix_data[i])
            else:
                print(">>> No data in Matrix!")
        else:
            print(">>> Not object specified!")

    # multiply two Matrix objects together
    @staticmethod
    def mat_mul(A, B):
        C = Matrix(A.rows, B.cols, mat_type="zeros")
        if A.cols != B.rows:
            print(">>> The two Matrices are not compatible with this operation!")
            return None
        else:
            for i in range(A.rows):
                for j in range(B.cols):
                    for k in range(B.rows):
                        C.matrix_data[i][j] += A.matrix_data[i][k] * B.matrix_data[k][j]
            return C

    @staticmethod
    def mat_add(A, B):
        C = Matrix(A.rows, B.cols, mat_type="zeros")
        # if A.cols != B.rows:
        #     print(">>> The two Matrices are not compatible with this operation!")
        #     return None
        # else:
        for i in range(A.rows):
            for j in range(B.cols):
                C.matrix_data[i][j] += A.matrix_data[i][j] + B.matrix_data[i][j]
        return C

    # make a matrix object with all zeros in it
    def mat_type(self, type):
        if type == "random":
            self.randomize()
        if type == "zeros":
            self.zeros()
        if type == "gauss_random":
            self.gauss_random()

    # apply a function to every element of the matrix (for activation function)
    @staticmethod
    def apply_func(A, func):
        B = Matrix(A.rows, A.cols, mat_type="zeros")
        for i in range(A.rows):
            for j in range(A.cols):
                B.matrix_data[i][j] += func(A.matrix_data[i][j])
        return B

    def gauss_random(self):
        self.matrix_data = [[random.gauss(0, self.cols ** (-1 / 2)) for x in range(self.cols)] for y in range(self.rows)]

    def randomize(self):
        self.matrix_data = [[random.uniform(-1, 1) for x in range(self.cols)] for y in range(self.rows)]

    def zeros(self):
        self.matrix_data = [[0 for x in range(self.cols)] for y in range(self.rows)]

    @staticmethod
    def copy(A):
        B = Matrix(A.rows, A.cols, mat_type="zeros")
        for i in range(A.rows):
            for j in range(A.cols):
                B.matrix_data[i][j] += A.matrix_data[i][j]
        return B

    @staticmethod
    def createVector(data):
        new_Vector = Matrix(len(data), 1, mat_type="zeros")
        for i in range(new_Vector.rows):
            new_Vector.matrix_data[i][0] = data[i]
        return new_Vector

    @staticmethod
    def vector_add(A, other):
        new_Vector = Matrix(A.rows, 1, mat_type="zeros")
        if A.rows is not other.rows:
            print(">>> Cannot add. Objects are not Vectors or have different dimensions!")
            return None
        for i in range(A.rows):
            new_Vector.matrix_data[i][0] += other.matrix_data[i][0] + A.matrix_data[i][0]
        return new_Vector

    @staticmethod
    def vector_sub(A, other):
        new_Vector = Matrix(A.rows, 1, mat_type="zeros")
        if A.rows is not other.rows:
            print(">>> Cannot substract. Objects are not Vectors or have different dimensions!")
            return None
        for i in range(A.rows):
            new_Vector.matrix_data[i][0] += other.matrix_data[i][0] - A.matrix_data[i][0]
        return new_Vector

    @staticmethod
    def vector_mult(A, other):
        new_Vector = Matrix(A.rows, 1, mat_type="zeros")
        if A.rows is not other.rows:
            print(">>> Cannot multiply. Objects are not Vectors or have different dimensions!")
            return None
        for i in range(A.rows):
            new_Vector.matrix_data[i][0] += other.matrix_data[i][0] * A.matrix_data[i][0]
        return new_Vector

    @staticmethod
    def add(A, x):
        B = Matrix(A.rows, A.cols, mat_type="zeros")
        for i in range(A.rows):
            for j in range(A.cols):
                B.matrix_data[i][j] += A.matrix_data[i][j] + x
        return B

    @staticmethod
    def mult(A, x):
        B = Matrix(A.rows, A.cols, mat_type="zeros")
        for i in range(A.rows):
            for j in range(A.cols):
                B.matrix_data[i][j] += A.matrix_data[i][j] * x
        return B

    @staticmethod
    def sub(A, x):
        B = Matrix(A.rows, A.cols, mat_type="zeros")
        for i in range(A.rows):
            for j in range(A.cols):
                B.matrix_data[i][j] += A.matrix_data[i][j] - x
        return B

    @staticmethod
    def soft_max(A):
        if A.cols is not 1:
            print(">>> Cannot soft-max Vector object!")
            return None
        else:
            B = Matrix(A.rows, A.cols, mat_type="zeros")
            sum = 0
            for item in A.matrix_data:
                sum += math.e ** item[0]
            for i in range(A.rows):
                B.matrix_data[i][0] = (math.e ** A.matrix_data[i][0]) / sum
            return B

    @staticmethod
    def transpose(A):
        new_rows = A.cols
        new_cols = A.rows
        new_Matrix = Matrix(new_rows, new_cols, mat_type="zeros")
        for i in range(A.rows):
            for j in range(A.cols):
                new_Matrix.matrix_data[j][i] = A.matrix_data[i][j]
        return new_Matrix
