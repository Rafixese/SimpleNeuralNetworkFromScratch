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
        
    # def num_of_weights_per_one_neuron(self):
    #     return len(self.weights) // self.neurons_num
    
    # def weights_arr_for_specific_neuron(self, neuron_num):
    #     arr = []
    #     for i in range(neuron_num * self.num_of_weights_per_one_neuron(), 
    #                    (neuron_num+1) * self.num_of_weights_per_one_neuron()):
    #         arr.append(i)
    #     return arr
    
class Model:
    
    def __init__(self):
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    # def left_neur_index_by_weight_number(self, weight_number, layer_number):
    #     return weight_number % self.layers[layer_number].num_of_weights_per_one_neuron()
        
    def predict(self, x):
        #########################################
        #      Obliczenie dla modelu n x m      #
        #########################################
        # pierwszymi wartościami będą wartości wejścia
        # neuron_values = [x]
        
        # # iterujemy po wszystkich warstwach
        # for layer_num in range(len(self.layers)):
        #     # deklarujemy tablicę w której będziemy umieszczać wartości neuronów z aktualnej warstwy
        #     layer_values = []
        #     # iterujemy po neuronach w danej warstwie
        #     for neur_num in range(self.layers[layer_num].neurons_num):
        #         # chcemy obliczyć wartość neuronu
        #         value = 0
        #         # iterujemy po wagach dla danego neuronu, 
        #         for weight_num in self.layers[layer_num].weights_arr_for_specific_neuron(neur_num):
        #             # do wartości neuronu dodajemy iloczyn danej wagi i wartości połączonego z nią neuronu z warstwy po lewej
        #             value += self.layers[layer_num].weights[weight_num] * neuron_values[layer_num][self.left_neur_index_by_weight_number(weight_num, layer_num)]
        #         # jeżeli mamy już sumę wszystkich wag * połączonych do nich neuronów dodajemy ją do tymczasowej tablicy (pomnożoną jeszcze o wartość biasu)
        #         layer_values.append(NeuralNetworkUtils.sigmoid(value * self.layers[layer_num].biases[neur_num]))
        #     # gdy zgromadzimy wszystkie wartości neuronów danej warstwy dodajemy tablice do ogólnej tablicy
        #     # ze wszystkimi wartościami neuronów w modelu
        #     neuron_values.append(layer_values.copy())

        # zwracamy wynik przewidywania przez nasz model który będzie znajdował się w ostatniej warstwie w pierwszym indexie
        # oraz wszystkie wyniki poszczególnych neuronów
        # return neuron_values[-1][0], neuron_values
        
        #########################################
        #       Obliczenie dla modelu 2x1       #
        #########################################
        
        h1 = NeuralNetworkUtils.sigmoid( self.layers[0].weights[0] * x[0] + self.layers[0].weights[1] * x[1] + self.layers[0].biases[0])
        h2 = NeuralNetworkUtils.sigmoid( self.layers[0].weights[2] * x[0] + self.layers[0].weights[3] * x[1] + self.layers[0].biases[1])
        o1 = NeuralNetworkUtils.sigmoid( self.layers[1].weights[0] * h1 + self.layers[1].weights[1] * h2 + self.layers[1].biases[0])
        return h1, h2, o1
        
    
    def train(self, x_train, y_train, learn_rate = 0.1, epochs = 100):
        #iterujemy po epokach
        for epoch in range(epochs):
            # iterujemy po zestawie danych wyciągając z niego po jednym rekordzie dla x i y
            for x, y in zip(x_train, y_train):
                # przepowiadamy za pomocą modelu jakie wg niego powinno być y, dostajemy też poszczególne wartości neuronów
                h1, h2, o1 = self.predict(x)
                # obliczamy pochodną z mse do późniejszej zmiany wag
                deriv_mse = -2 * (y - o1)
                
                #########################################
                #      Obliczenie dla modelu n x m      #
                #########################################
                
                # # tutaj będziemy trzymać wyniki
                # delta_weights = []
                # delta_biases = []
                # delta_neurons = []
                
                # # idziemy po warstwach od tyłu
                # for layer_num in reversed(range(len(self.layers))):
                    
                #     curr_delta_weights = []
                #     curr_delta_biases = []
                #     curr_delta_neurons = []
                    
                #     # dla każdej warstwy iterujemy po neuronach
                #     for neuron_num in range(self.layers[layer_num].neurons_num):
                #         # wyliczamy pochodną od neuronu
                #         deriv_neuron =  NeuralNetworkUtils.deriv_sigmoid(neuron_values[layer_num+1][neuron_num])
                #         # dla każdego neuronu wyliczamy deltę wag przyległych do tego neuronu (d_w = <wartość neuronu po lewej stronie wagi> * <pochodna z danego neuronu>)
                #         for weight_num in self.layers[layer_num].weights_arr_for_specific_neuron(neuron_num):
                #             curr_delta_weights.append( neuron_values[layer_num][ self.left_neur_index_by_weight_number(weight_num, layer_num)] * deriv_neuron)
                #         # dla każdego neuronu wyliczamy delte biasu
                #         curr_delta_biases.append(deriv_neuron)
                   
                #         #   Wyliczenie delt dla neuronów z poprzedniej warstwy
                #         #   ??????????????????????????????????????????????????
                #         #   dla modelu 2x1 obliczenie neuronów ukrytej warstwy jest proste
                #         #   nie mam pomysłu co zrobić gdy trzeba obliczyć delty pomiędzy warstwami np 3 i 2 neurony
                
                #########################################
                #       Obliczenie dla modelu 2x1       #
                #########################################
                
                d_w5 = h1 * NeuralNetworkUtils.deriv_sigmoid(o1)
                d_w6 = h2 * NeuralNetworkUtils.deriv_sigmoid(o1)
                d_b3 = NeuralNetworkUtils.deriv_sigmoid(o1)
                d_h1 = self.layers[1].weights[0] * NeuralNetworkUtils.deriv_sigmoid(o1)
                d_h2 = self.layers[1].weights[1] * NeuralNetworkUtils.deriv_sigmoid(o1)
                
                d_w1 = x[0] * NeuralNetworkUtils.deriv_sigmoid(h1)
                d_w2 = x[1] * NeuralNetworkUtils.deriv_sigmoid(h1)
                d_b1 = NeuralNetworkUtils.deriv_sigmoid(h1)
                
                d_w3 = x[0] * NeuralNetworkUtils.deriv_sigmoid(h2)
                d_w4 = x[1] * NeuralNetworkUtils.deriv_sigmoid(h2)
                d_b2 = NeuralNetworkUtils.deriv_sigmoid(h2)
                
                self.layers[0].weights[0] -= deriv_mse * learn_rate * d_h1 * d_w1
                self.layers[0].weights[1] -= deriv_mse * learn_rate * d_h1 * d_w2
                self.layers[0].biases[0] -= deriv_mse * learn_rate * d_h1 * d_b1
                
                self.layers[0].weights[2] -= deriv_mse * learn_rate * d_h2 * d_w3
                self.layers[0].weights[3] -= deriv_mse * learn_rate * d_h2 * d_w4
                self.layers[0].biases[1] -= deriv_mse * learn_rate * d_h2 * d_b2
                
                self.layers[1].weights[0] -= deriv_mse * learn_rate * d_w5
                self.layers[1].weights[1] -= deriv_mse * learn_rate * d_w6
                self.layers[1].biases[0] -= deriv_mse * learn_rate * d_b3
                
            if epoch % 10 == 0:
                y_pred_all = np.apply_along_axis(self.predict, 1, x_train)[:,2]
                loss = NeuralNetworkUtils.mse(y_train, y_pred_all)
                print("Epoch", epoch, "Loss:", loss)
                print(y_pred_all > 0.5)
    
#%% Model build

model = Model()
model.addLayer(Layer(2,2))
model.addLayer(Layer(2,1))

#%% Train
X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]
Y = [0,0,0,1]

model.train(X, Y, epochs = 1000, learn_rate=0.1)

y_pred_all = np.apply_along_axis(model.predict, 1, X)





