#Solution to question 2.

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt

def sigmoid(x):
	return (1.0 / (1.0 + np.exp(-x)))	

def binary(x):
	return 1. if x >= 0.5 else 0.

def weightUpdate(X,Y,theta_1,theta_2,number_epochs,learning_rate):
	for epoch in range(number_epochs):
	
		sys.stdout.write("Training progress: [%d / %d]	\r" % (epoch+1, number_epochs))
		sys.stdout.flush()

		x = np.hstack((np.ones((100,1)), X))
		for i in range(n_samples):
			#Forward pass
			inputs = x[i,:].reshape(3,1)
			hidden_layer_output = sigmoid(theta_1.dot(inputs) )
			output = sigmoid(theta_2.T.dot(hidden_layer_output))
			#Backpropagation
			grad_2 = (Y[i] - output)*(1-output)*output*hidden_layer_output
			grad_1 = np.dot((grad_2*(theta_2)*(1-hidden_layer_output)),inputs.T)

			theta_1 = theta_1 + learning_rate*grad_1
			theta_2 = theta_2 + learning_rate*grad_2

	return theta_1, theta_2

def predict(X,Y,theta_1,theta_2,n_samples):

	x = np.hstack((np.ones((100,1)),X))

	hidden_layer_output = sigmoid(theta_1.dot(x.T))
	output = sigmoid(theta_2.T.dot(hidden_layer_output)).T
	number_correct = 0
	
	for i in range(n_samples):
		output[i] = binary(output[i])
		if(output[i] == Y[i]):
			number_correct+=1

	accuracy = float(number_correct) / float(n_samples) * 100

	return output, accuracy

def plotHyperplanes(X, Y, theta, epoch):
	colors = ['red','blue']

	fig, ax = plt.subplots(figsize=(8,8))
	ax.scatter(X[:,0],X[:,1], c=Y,cmap=matplotlib.colors.ListedColormap(colors))

	x = np.hstack((np.ones((100,1)), X))
	for i in range(theta.shape[0]):
		y1 = (-1. / theta[i,2]) * (theta[i,0])
		y2 = (-1. / theta[i,2]) * (theta[i,0] + theta[i,1])

		
		ax.plot([0,1], [y1,y2], 'g--')
	
	ax.set_xlim([-0.5,1.5])
	ax.set_ylim([-0.5,1.5])
	plt.xlabel(r'$x_1$')
	plt.ylabel(r'$x_2$')
	plt.title("Hyperplanes Corresponding to Hidden Nodes at epoch %d" % epoch)
	plt.show()



#Generate 100 random samples on the unit square
n_samples = 100
X = np.random.uniform(0,1,size=(n_samples,2))

#Given parameters for the circle on the unit square
a = 0.5
b = 0.6
r = 0.4
Y = (((X[:,0] - a)**2 + (X[:,1] - b)**2) < r**2)

#Test set of 100 samples
X_test = np.random.uniform(0,1,size=(n_samples,2))
Y_test = (((X_test[:,0] - a)**2 + (X_test[:,1] - b)**2) < r**2)

#Hyperparameters
hidden_nodes = 10
output_nodes = 1
learning_rate = 1
number_epochs = 200

#theta_1, theta_2 = weightUpdate(X,Y,theta_1,theta_2,number_epochs,learning_rate)

colors = ['red','blue']
plt.figure(figsize=(8,8))
plt.scatter(X[:,0],X[:,1],c=Y,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title("Training Set")
plt.show()

#output, accuracy = predict(X,Y,theta_1,theta_2,n_samples)

theta_1_best = np.ones((hidden_nodes,3))
theta_2_best = np.ones((hidden_nodes,output_nodes))
best_accuracy = 0.
best_epoch = 0

iterations = []
accuracies = []

#Train the neural network on a gradually increasing number of epochs
for epochs in range(number_epochs):

	#Randomly initialize the weights corresponding to each layer
	theta_1 = np.random.rand(hidden_nodes,3)
	theta_2 = np.random.rand(hidden_nodes,output_nodes)

	#Update weights against the training set
	theta_1, theta_2 = weightUpdate(X,Y,theta_1,theta_2,epochs,learning_rate)

	

	#Get the accuracy on the test set
	output, accuracy = predict(X_test,Y_test,theta_1,theta_2,n_samples)
	
	#Plot hyperplanes at arbitrary epochs over the test set
	if(epochs == 0 or epochs == 100):
		plotHyperplanes(X_test,output,theta_1, epochs)

	iterations.append(epochs)
	accuracies.append(accuracy)

	#Store the best weights and corresponding accuracy 
	if(accuracy > best_accuracy):
		best_accuracy = accuracy
		best_epoch = epochs
		theta_1_best = theta_1
		theta_2_best = theta_2

#Plot accuracies as a function of the number of epochs trained
plt.plot(iterations, accuracies)
plt.title("Accuracy on Test Set vs. Number of Epochs Trained")
plt.xlabel("# epochs")
plt.ylabel("Accuracy (%)")
plt.show()

#Predict the classes of the test set using the best network weights
output, accuracy = predict(X_test,Y_test,theta_1_best,theta_2_best,n_samples)

#Plot the hyperplanes corresponding to the hidden nodes over the test set
#at the best found setting of weights
plotHyperplanes(X_test, output, theta_1_best, best_epoch)

#Plot the test set and the correct classes
plt.figure(figsize=(8,8))
plt.scatter(X_test[:,0],X_test[:,1],c=Y_test,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title("Test Set")
plt.show()

print "\n Best accuracy: ", accuracy, "%"