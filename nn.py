from functions import *
import numpy as np
import math
from sklearn.datasets import load_iris
#import random
random_state=None
seed = None if random_state is None else int(random_state)
rng = np.random.default_rng(seed=seed)

class NeuralNetwork:
    def __init__(self, learning_rate, epochs, size_input, neuron_hidden, size_output):
        self.learn_rate = learning_rate
        self.epoch = epochs
        self.size_input = size_input
        self.size_output = size_output
        self.num_neuron_hidden = neuron_hidden
        print("--------------------------------------------------")
        print("Neural network: ")
        print("--------------------------------------------------")
        print("Input: ", self.size_input)
        print("Hidden: ", self.num_neuron_hidden)
        print("Output: ", self.size_output)

        #lower_w1, upper_w1 = -(math.sqrt(6.0) / math.sqrt(size_input + self.size_input)), (math.sqrt(6.0) / math.sqrt(size_input + self.size_input))
        #self.w1 = np.random.uniform(lower_w1, upper_w1, size=(self.num_neuron_hidden, size_input))
        self.w1 = np.random.randn(self.num_neuron_hidden, size_input) * 0.01
        print("w1: ", self.w1)
        #self.w2 = np.random.uniform(lower_w1, upper_w1, size=(self.size_output, self.num_neuron_hidden))
        self.w2 = np.random.randn(self.size_output, self.num_neuron_hidden) * 0.01 
        print("w2: ", self.w2)
        #self.b1 = np.random.uniform(lower_w1, upper_w1, size=(self.num_neuron_hidden))
        #self.b2 = np.random.uniform(lower_w1, upper_w1, size=(self.size_output))
        self.b1 = np.zeros((self.num_neuron_hidden))
        self.b2 = np.zeros((self.size_output))
        print("--------------------------------------------------")


    def feedforward(self, x):
        self.input_data = x
        #print("-------------------------------------------------------------------------") 
        #print("x.shape: ", self.input_data.shape)
        #print("self.w1.shape: ", self.w1.shape) 
        #print("self.b1.shape: ", self.b1.shape) 
        self.z1 = np.dot(self.w1, x) + self.b1
        #print("self.z1.shape: ", self.z1.shape)
        self.sigmoid_hidden = sigmoid(self.z1)
        #print("self.sigmoid_hidden.shape: ", self.sigmoid_hidden.shape)
        #print("self.w2.shape: ", self.w2.shape) 
        #print("self.b2.shape: ", self.b2.shape) 
        self.z2 = np.dot(self.w2, self.sigmoid_hidden) + self.b2
        #print("self.z2.shape: ", self.z2.shape)
        self.sigmoid_output = sigmoid(self.z2)
        #print("self.sigmoid_output.shape: ", self.sigmoid_output.shape)
        #print("-------------------------------------------------------------------------") 
        return self.sigmoid_output 


    def backpropogation(self, x, y, i):
        delta = mse_derivative(y, x) * deriv_sigmoid(self.z2)         

        grad_w2 = delta  
        grad_b2 = delta 
        grad_w2 = np.outer(grad_w2, self.sigmoid_hidden.T) 

        self.w2 -= self.learn_rate * grad_w2 
        self.b2 -= self.learn_rate * grad_b2

        delta_input = (delta @ self.w2) * deriv_sigmoid(self.z1)
        grad_w1 = np.outer(delta_input, self.input_data.T)
        grad_b1 = delta_input 
        
        self.w1 -= self.learn_rate * grad_w1 
        self.b1 -= self.learn_rate * grad_b1

    def train(self, x, y, all_train):
        #print("all: ", all_train)
        size_data = len(x)
        all_pred = []
        batch_size = 40
        #print("all: ", all_train[:batch])
        #print("num: iter: ", round(size_data / batch))
        num_batch = round(size_data / batch_size) 
        #print("all each 1: ", all_train[num_batch * 1:batch])
        #print("all each 2: ", all_train[num_batch * 2:batch])


        for ep in range(self.epoch):
            rng.shuffle(all_train)
            for index in range(num_batch):
                stop = index + batch_size

                x_batch, y_batch = all_train[index:stop, :-1], all_train[index:stop, -1:]
                for i in range(len(x_batch)):
                    #print("x_batch: ", x_batch[i][0][0], "y_batch: ", y_batch[i][0][0])
                    pred = self.feedforward(x_batch[i][0][0])
                #print("pred: ", pred)
                #print("y: ", y[index])
                    all_pred.append(np.array(pred))
                    self.backpropogation(pred, y_batch[i][0][0], index)
                    error = mse_loss(pred, y_batch[i][0][0]) 
            


            all_pred = []
            #print("self.w2 ", self.w2)


            if ep % 10 == 0:
                print("--------------------")
                print("epoch: ", ep)
                print("error", error)
                print("setosa: ", network.feedforward(np.array([5.1,3.5,1.4,0.2])))
                print("setosa argmax: ", np.argmax(np.asarray(network.feedforward(np.array([5.1,3.5,1.4,0.2])))))
                print("versicolor argmax: ", np.argmax(np.asarray(network.feedforward(np.array([5.5,2.5,4.0,1.3])))))
                print("versicolor: ", network.feedforward(np.array([5.5,2.5,4.0,1.3])))
                print("virginica argmax: ", np.argmax(np.asarray(network.feedforward(np.array([5.9,3.0,5.1,1.8])))))
                print("virginica: ", network.feedforward(np.array([5.9,3.0,5.1,1.8])))


# Load the Iris dataset
iris = load_iris()

# Access the features and target variable
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)
#print("X", X)
#print("y", y)
iris_array = []
for i in range(len(iris.target)):
    iris_sentosa = [1, 0, 0]
    iris_versicolor = [0, 1, 0]
    iris_virginica = [0, 0, 1]
    if iris.target[i] == 0:
        iris_array.append(iris_sentosa)
    if iris.target[i] == 1:
        iris_array.append(iris_versicolor)
    if iris.target[i] == 2:
        iris_array.append(iris_virginica)     
    #iris_array.append(iris_arr_i[iris.target[i]])


print(iris_array)
print(len(iris_array))
iris_array = np.array(iris_array)

x_data = np.array([
    [1,0,0,1],
    [1,0,0,0],
    [0,0,0,0],
    [0,1,1,0],
    [0,0,0,1],
    [1,1,1,1],
    [0,0,1,1],
    [1,1,1,0]
])

y_data = np.array([
    [1,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
])

network = NeuralNetwork(0.1, 100, 4, 5, 3)


#all_train = np.array[X, iris_array]
n_obs = X.shape[0]
#all_train = #np.c_[X.reshape(n_obs, -1), iris_array.reshape(n_obs, 1)]
#all_train = np.array([X, iris_array])
all_train = []
#elem = [X[149],iris_array[149]]
for i in range(len(X)):
    elem = [[X[i]],[iris_array[i]]]
    all_train.append(np.array(elem, dtype=object))

#print("all train: ", all_train)
network.train(X, iris_array, np.array(all_train))



#print("other: ", network.feedforward([1,1,0,1]))
#print("other: ", network.feedforward([1,1,1,1]))
#print("other: ", network.feedforward([0,0,0,1]))



