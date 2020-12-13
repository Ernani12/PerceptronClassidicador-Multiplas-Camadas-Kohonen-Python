from __future__ import print_function
import matplotlib,sys
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.function_base import append
from numpy import array



W = [0] *3



def predict(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w 
	return 1.0 if activation>=0.0 else 0.0

def Line():
	x = np.linspace(2.8,0)
	x2 = np.linspace(0.8,0)
	y =(W[0]*x+W[1]*x2)+ (-1)*W[2]

	plt.plot(x, y, '-r', label='y=W1*x1+W2*x2+ W3')
	plt.xlabel('x', color='#1C2833')
	plt.ylabel('y', color='#1C2833')




def plot(matrix,weights=None,title="Prediction Matrix"):

	

	if len(matrix[0])==4: # if 2D inputs, excluding bias and ys
		fig,ax = plt.subplots()
		ax.set_title(title)
		ax.set_xlabel("x1")
		ax.set_ylabel("x2")

		if weights!=None:
			map_min=0
			map_max=12
			y_res=0.5
			x_res=0.5
			ys=np.arange(map_min,map_max,y_res)
			xs=np.arange(map_min,map_max,x_res)
			zs=[]
			for cur_y in np.arange(map_min,map_max,y_res):
				for cur_x in np.arange(map_min,map_max,x_res):
					zs.append(predict([1.0,cur_x,cur_y],weights))
			xs,ys=np.meshgrid(xs,ys)
			zs=np.array(zs)
			zs = zs.reshape(xs.shape)
			#cp=plt.contourf(xs,ys,zs,levels=[10, 30, 50],colors=('b','r'),alpha=1)
            

		c1_data=[[],[]]
		c0_data=[[],[]]
		for i in range(len(matrix)):
			cur_i1 = matrix[i][1]
			cur_i2 = matrix[i][2]
			cur_y  = matrix[i][-1]
			if cur_y==1:
				c1_data[0].append(cur_i1)
				c1_data[1].append(cur_i2)
			else:
				c0_data[0].append(cur_i1)
				c0_data[1].append(cur_i2)
        
		
         
		
		plt.xticks(np.arange(0,11,1))
		plt.yticks(np.arange(0,11,1))
		plt.xlim(0,11)
		plt.ylim(0,11)
	
		
		

		c0s = plt.scatter(c0_data[0],c0_data[1],s=50.0,c='r',label='Class -1')
		c1s = plt.scatter(c1_data[0],c1_data[1],s=50.0,c='b',label='Class 1')

		plt.legend(fontsize=10,loc=4)
	
		return
	
	print("Matrix dimensions not covered.")

# each matrix row: up to last row = inputs, last row = y (classification)
def accuracy(matrix,weights):
	num_correct = 0.0
	preds       = []
	for i in range(len(matrix)):
		pred   = predict(matrix[i][:-1],weights) # get predicted classification
		preds.append(pred)
		if pred==matrix[i][-1]: num_correct+=1.0 
	print("Predictions:",preds)
	return num_correct/float(len(matrix))

# each matrix row: up to last row = inputs, last row = y (classification)
def train_weights(matrix,weights,nb_epoch=10,l_rate=0.02,do_plot=True,stop_early=True,verbose=True):
	for epoch in range(nb_epoch):
	
		cur_acc = accuracy(matrix,weights)
		print("\nEpoch %d \nWeights: "%epoch,weights)
		print("Accuracy: ",cur_acc)
		
		if cur_acc==1.0 and stop_early: break 
		#if do_plot and len(matrix[0])==4: plot(matrix,weights) # if 2D inputs, excluding bias
		if do_plot: plot(matrix,weights,title="Epoch %d"%epoch)
		j=int
		for i in range(len(matrix)):
			prediction = predict(matrix[i][:-1],weights) # get predicted classificaion
			error      = matrix[i][-1]-prediction		 # get error from real classification
			if verbose: sys.stdout.write("Training on data at index %d...\n"%(i))
			
			for j in range(len(weights)): 				 # calculate new weight for each node
				if verbose: sys.stdout.write("\tWeight[%d]: %0.5f --> "%(j,weights[j]))
                  
				# do meu treinamento peso anterior
				weights[j] = weights[j]+(l_rate*error*matrix[i][j]) 
				W[j]=(weights[j])			
				if verbose: sys.stdout.write("%0.5f\n"%(weights[j]))
            	
	plot(matrix,weights,title="Resultados de Classificacao")         

	return weights 

def main():

	nb_epoch		= 10  # contador de epoca onde vai convergir
	l_rate  		= 0.03# constante aprendizado
	plot_each_epoch	= False
	stop_early 		= False


			# 	Bias(ouTeta) 	x1 		x2 		y
			# valores de x1 e x2 desejados e  saida y
	matrix = [		[1.00,	1.00,	1.00,	1.0],
					[1.00,	9.4,	6.4,	-1.0],
					[1.00,	2.5,	2.1,	1.0],
					[1.00,	8.0,	7.7,	-1.0],
					[1.00,	0.5,	2.2,	1.0],
					[1.00,	7.9,	8.4,	-1.0],
					[1.00,	7.0,	7.0,	-1.0],
					[1.00,	2.8,	0.8,	1.0],
					[1.00,	1.2,	3.0,	1.0],
					[1.00,	7.8,	6.1,	-1.0]]
	weights= [	 0.75, 0.5, -0.6		] # initial weights specified in problem

	

	train_weights(matrix,weights=weights,nb_epoch=nb_epoch,l_rate=l_rate,do_plot=plot_each_epoch,stop_early=stop_early)
	Line()
	plt.show()

if __name__ == '__main__':
	main()
print(*W) 