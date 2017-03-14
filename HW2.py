import csv, sys
import numpy as np

categories = {}

def binary(x):
	return 1. if x >= 0.5 else -1.

def sigmoid(x):
	if(x < -1e8):
		return 0.
	elif(x > 1e8):
		return 1.

	return (1.0 / (1.0 + np.exp(-x)))	


def toCategorical(index, in_string):
	global categories
	if(index in categories):
		if(in_string in categories[index]):
			return categories[index][in_string]
		else:
			category = max(categories[index].values()) + 1
			categories[index][in_string] = category
			return category
	else:
		categories[index] = {in_string : 0}
		return 0

def binaryClass(in_string):
	if(in_string == "yes"):
		return 1.
	elif(in_string == "no"):
		return -1.

def processDataset(csv_file, samples, start):

	with open(csv_file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=';')
		feature_length = len(reader.next()) - 1 #skip the first line, get number of features

		X = np.empty((samples, feature_length))
		Y = np.empty((samples, 1))

		for i in range(start):
			reader.next()

		for i in range(samples):
			row = reader.next()

			for j in range(feature_length):
				if(row[j].lstrip('-').isdigit()):
					X[i,j] = float(row[j])
				elif (type(row[j]) == type("string")):
					X[i,j] = toCategorical(j, row[j])
			
			Y[i] = binaryClass(row[len(row) - 1])

		return (X, Y)

def weightUpdateSign(W, X, Y, learning_rate, iterations):
	for step in range(iterations):

		sys.stdout.write("Training progress: [%d / %d]	\r" % (step+1, iterations))
		sys.stdout.flush()

		for i in range(X.shape[0]):
			if(np.sign(W.dot(X[i,:])) != Y[i]):
				W = W + learning_rate * Y[i] * X[i,:]
				#print learning_rate*Y[i]*X[i,:]
	
	return W

def weightUpdateSigmoid(W, X, Y, learning_rate, iterations):
	for step in range(iterations):

		sys.stdout.write("Training progress: [%d / %d]	\r" % (step+1, iterations))
		sys.stdout.flush()

		for i in range(X.shape[0]):
			output = sigmoid(W.dot(X[i,:]))
			#if(binary(output) != Y[i]):
			W = W +learning_rate*(Y[i] - output)*(1 - output)*output*X[i,:]
		
	return W

def predict(W, X, Y):
	number_correct = 0
	for i in range(X.shape[0]):
		if np.sign(W.dot(X[i,:])) == Y[i]:
			number_correct += 1

	accuracy = float(number_correct) / X.shape[0] * 100
	return accuracy

if __name__ == "__main__":
	
	number_samples = 4521 #Cheaper to just hardcode it for this single dataset

	X_train, Y_train = processDataset('bank.csv', samples = number_samples/2, start = 0)
	X_test, Y_test = processDataset('bank.csv', samples = number_samples / 2, start = number_samples / 2 + 1)
	W = np.ones(X_train.shape[1])
	
	learning_rate = 0.02

	print("Training...")
	W = weightUpdateSigmoid(W, X_train, Y_train, learning_rate, iterations = 80)
	print("\nDone.")
	print("Accuracy on training set: "), predict(W, X_train,Y_train), "%"
	print("Accuracy on test set: "), predict(W, X_test, Y_test), "%"

