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
        return x * (1 - x)
    
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
        
    def num_of_weights_per_one_neuron(self):
        return len(self.weights) // self.neurons_num
    
    def weights_arr_for_specific_neuron(self, neuron_num):
        arr = []
        for i in range(neuron_num * self.num_of_weights_per_one_neuron(), 
                       (neuron_num+1) * self.num_of_weights_per_one_neuron()):
            arr.append(i)
        return arr
    
class Model:
    
    def __init__(self):
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    def left_neur_index_by_weight_number(self, weight_number, layer_number):
        return weight_number % self.layers[layer_number].num_of_weights_per_one_neuron()
        
    def predict(self, x):
        # pierwszymi wartościami będą wartości wejścia
        neuron_values = [x]
        
        # iterujemy po wszystkich warstwach
        for layer_num in range(len(self.layers)):
            # deklarujemy tablicę w której będziemy umieszczać wartości neuronów z aktualnej warstwy
            layer_values = []
            # iterujemy po neuronach w danej warstwie
            for neur_num in range(self.layers[layer_num].neurons_num):
                # chcemy obliczyć wartość neuronu
                value = 0
                # iterujemy po wagach dla danego neuronu, 
                for weight_num in self.layers[layer_num].weights_arr_for_specific_neuron(neur_num):
                    # do wartości neuronu dodajemy iloczyn danej wagi i wartości połączonego z nią neuronu z warstwy po lewej
                    value += self.layers[layer_num].weights[weight_num] * neuron_values[layer_num-1][self.left_neur_index_by_weight_number(weight_num, layer_num)]
                # jeżeli mamy już sumę wszystkich wag * połączonych do nich neuronów dodajemy ją do tymczasowej tablicy
                layer_values.append(NeuralNetworkUtils.sigmoid(value))
            # gdy zgromadzimy wszystkie wartości neuronów danej warstwy dodajemy tablice do ogólnej tablicy
            # ze wszystkimi wartościami neuronów w modelu
            neuron_values.append(layer_values.copy())

        # zwracamy wynik przewidywania przez nasz model który będzie znajdował się w ostatniej warstwie w pierwszym indexie
        # oraz wszystkie wyniki poszczególnych neuronów
        return neuron_values[-1][0], neuron_values
    
    def predict_array(self, x):
        arr = []
        for row in x:
            arr.append(self.predict(row))
        return arr
    
    def train(self, x_train, y_train, learn_rate = 0.1, epochs = 100):
        #iterujemy po epokach
        for epoch in range(epochs):
            # iterujemy po zestawie danych wyciągając z niego po jednym rekordzie dla x i y
            for x, y in zip(x_train, y_train):
                # przepowiadamy za pomocą modelu jakie wg niego powinno być y, dostajemy też poszczególne wartości neuronów
                y_pred, neuron_values = self.predict(x)
                # obliczamy pochodną z mse do późniejszej zmiany wag
                deriv_mse = -2 * (y - y_pred)
                
                delta_wages = []
                delta_biases = []
                delta_neurons = []
                
                pass
                
            
    
#%% Test

model = Model()
model.addLayer(Layer(2,2))
model.addLayer(Layer(2,1))

#%% Test 2
X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]
Y = [0,0,0,1]
predict, neuron_values = model.predict(X[1])
mse = NeuralNetworkUtils.mse(Y, predict)

model.train(X, Y, epochs = 5)





