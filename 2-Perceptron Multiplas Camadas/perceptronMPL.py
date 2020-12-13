import numpy as np
import matplotlib.pyplot as plt
import math



def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))  # funcao da  rede sigmoid

def sigmoid_der(x):
	return x*(1.0 - x)     # regra delta

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l=len(self.inputs)
        self.li=len(self.inputs[0])

        self.wi=np.random.random((self.li, self.l)) #pesos aletoriso mudam saidas treinamento
        self.wh=np.random.random((self.l, 1))

    def think(self, inp):
        s1=sigmoid(np.dot(inp, self.wi))
        s2=sigmoid(np.dot(s1, self.wh))
        return s2

    def train(self, inputs,outputs, it):
        for i in range(it):
            l0=inputs
            l1=sigmoid(np.dot(l0, self.wi))   #camada saida
            l2=sigmoid(np.dot(l1, self.wh))  # camada oculta

            l2_err=outputs - l2
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))

            l1_err=np.dot(l2_delta, self.wh.T)
            l1_delta=np.multiply(l1_err, sigmoid_der(l1))

            self.wh+=np.dot(l1.T, l2_delta)
            self.wi+=np.dot(l0.T, l1_delta)

inputs=np.array([[0,0], [1,0], [0,1], [1,1] ])
outputs=np.array([ [0], [1],[1],[0] ])

n=NN(inputs)
print ("Before training :")
print(n.think(inputs))
print ("After  training :")
n.train(inputs, outputs, 1400)
print(n.think(inputs))





