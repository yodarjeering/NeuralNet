import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from collections import OrderedDict
from optimizer import *
from function import *
from neuralnet import *


(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train = x_train / 255
num_classes = 10
t_train = np_utils.to_categorical(t_train, num_classes)


x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test = x_test / 255
num_classes = 10
t_test = np_utils.to_categorical(t_test, num_classes)


optimizers = {}
optimizers['SGD'] = SGD()
optimizers['RMSprop'] = RMSprop()
optimizers['AMSGrad'] = AMSGrad()
optimizers['Adam'] = Adam()


train_size = x_train.shape[0]
batch_size = 128
epochs = 5000


networks = {}
train_loss = {}
train_accuracy = {}
for key in optimizers.keys():
    networks[key] = NeuralNet(
        input_size=784, hidden_size_list=[100,100],
        output_size=10)
    train_loss[key] = []
    train_accuracy[key] = []


for i in range(epochs):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        accuracy = networks[key].accuracy(x_batch, t_batch)
        train_accuracy[key].append(accuracy)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("-----------------------")
        print("epoch:",i)
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":",loss)


plt.plot(train_loss['SGD'])
plt.show()
plt.plot(train_accuracy['SGD'])
plt.show()


plt.plot(train_loss['RMSprop'])
plt.show()
plt.plot(train_accuracy['RMSprop'])
plt.show()


plt.plot(train_loss['Adam'])
plt.show()
plt.plot(train_accuracy['Adam'])
plt.show()


plt.plot(train_loss['AMSGrad'])
plt.show()
plt.plot(train_accuracy['AMSGrad'])
plt.show()
