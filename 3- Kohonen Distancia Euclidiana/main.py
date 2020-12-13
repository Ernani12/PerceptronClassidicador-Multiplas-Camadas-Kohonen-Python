from internal.network import *
from internal.data import *
from internal.plot import *



focals = [[1, 1],
	      [2.5, 6.4],
		  [2.5, 2.1],
		  [8, 7.7],
		  [0.5, 2.2],
		  [7.9, 8.4],
		  [7, 7],
		  [2.8, 0.8],
		  [1.2, 3],
	      [7.8, 6.1]]

data = GenerateData(focals)

network = Network(5, 2) #ao todo serao 10 interacoes separadas 
network.data = data

plot = Plot()
plot.Scatter(data)
plot.Scatter(network.Unpack())
plot.Show()

network.Train(10)

plot.Scatter(data)
plot.Scatter(network.Unpack())
plot.Show()