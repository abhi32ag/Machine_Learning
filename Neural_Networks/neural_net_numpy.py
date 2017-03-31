import numpy as np 



class NeuralNetwork():

	def __init__(self):
		# seed the random number generator 
		np.random.seed(1)

		# We model a single neuron, with 3 input connections and 1 output 
		# we assign random weights to the 3 x 1 matrix, values in range -1 to 1 
		# and mean 0 
		self.synaptic_weights = 2 * np.random.rando((3,1)) - 1

		
	


if "__name__" == "__main__":

	neural_network = NeuralNetwork()

	print('Random starting synaptic weights')
	print(neural_network.synaptic_weights)

	#In the training set we have four examples, each consists of 3 input values 
	# and 1 output value 
	training_set_inputs = np.array([0,0,1], [1,1,1], [1,0,1], [0,1,1])
	training_set_outputs = np.array([0,1,1,0]).T 

	# train the neural net with the training set
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print('New synaptic weights after training ')
	print(neural_network.synaptic_weights)

	#test the neural network 

	print('Considering new situation [1,0,0]')
	print(neural_network.predict(np.array([1,0,0])))

