#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:12:25 2020

@author: ganja
"""

#%% Imports

import numpy as np
import math

#%% Classes

class NeuralNetworkUtils:
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def deriv_sigmoid(x):
        return NeuralNetworkUtils.sigmoid(x) * (1 - NeuralNetworkUtils.sigmoid(x))
    
    @staticmethod
    def mse(y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)
        return np.mean((y - y_pred) ** 2)

class Layer:
    
    def __init__(self, input_neurons, neurons_num):
        self.weights = np.random.rand(input_neurons * neurons_num,)
        self.biases = np.random.rand(neurons_num,)
        self.neurons_num = neurons_num
    
class Model:
    
    def __init__(self):
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    def predict(self, x):
        if len(x) != self.layers[0].neurons_num:
            print("Wrong input array shape")
            return None
        
        prev_layer_neuron_values = x
        curr_values = []
        neuron_values = []
        
        for layer in self.layers:
            for neur_num in range(layer.neurons_num):
                value = 0
                for weight_num in range(neur_num * layer.neurons_num, (neur_num+1) * layer.neurons_num):
                    value += layer.weights[weight_num] * prev_layer_neuron_values[weight_num % layer.neurons_num]
                curr_values.append(NeuralNetworkUtils.sigmoid(value))
            prev_layer_neuron_values = curr_values.copy()
            neuron_values.append(curr_values.copy())
            curr_values.clear()
        
        return prev_layer_neuron_values[0], neuron_values
    
    def predict_array(self, x):
        arr = []
        for row in x:
            arr.append(self.predict(row))
        return arr
    
    def train(self, x_train, y_train, learn_rate, epochs):
        for x, y in zip(x_train, y_train):
            y_pred, neuron_values = self.predict(x)
            pass
            
            
    
#%% Test

layer1 = Layer(2,2)
output_layer = Layer(2,1)

model = Model()
model.addLayer(layer1)
model.addLayer(output_layer)

#%% Test 2
X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]
Y = [0,0,0,1]
predict, neuron_values = model.predict(X[0])
mse = NeuralNetworkUtils.mse(Y, predict)

#%%

for i in range(0, 10): print(9-i)










