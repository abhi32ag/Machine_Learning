import numpy as np 



class NeuralNetwork:
	


if "__name__" == "__main__":

	neural_network = NeuralNetwork()

	print('Random starting synaptic weights')
	print(neural_network.synaptic_weights)

	#In the training set we have four examples, each consists of 3 input values 
	# and 1 output value 
	training_set_inputs = np.array([0,0,1], [1,1,1], [1,0,1], [0,1,1])
	training_set_outputs = np.array([])