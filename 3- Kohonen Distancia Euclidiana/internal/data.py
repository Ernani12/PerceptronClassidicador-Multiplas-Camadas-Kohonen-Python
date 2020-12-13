from random import  gauss

def GenerateData(points, sigma=1):
	data = []
	for point in points:
		for i in range(1):   # passa pela intercoes necesarias pontos gerados mais sigma
			data.append({
				'x': gauss(point[0], sigma),     #distancia 
				'y': gauss(point[1], sigma)})
	return data
							   