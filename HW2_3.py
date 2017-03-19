#Solution to question 3.

import numpy as np
import csv, sys
import matplotlib
import matplotlib.pyplot as plt
import random

def sigmoid(x):
	return (1.0 / (1.0 + np.exp(-x)))	

samples = 200

X = np.empty((samples, 2))
Y = np.empty((samples, 2))

with open("foo.csv", 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		reader.next() #skip a line

		for i in range(samples):
			row = reader.next()

			for j in range(2):
				X[i,j] = row[j]
				Y[i,j] = row[j+2]

number_epochs = 20000
learning_rate = 2.2
accuracies = []
iteration = []
best_accuracy = -1.
best_iteration = -1
best_b1, best_w1, best_w_hat1 = (0.,0.,0.)
best_b2, best_w2, best_w_hat2 = (0.,0.,0.)
#for j in range(number_epochs):


b1 = random.random()  % 4 - 1
w1 = random.random()  % 4 - 1
w_hat1 = random.random()  % 4 - 1

b2 = random.random()  % 4 - 1
w2 =random.random()  % 4 - 1
w_hat2 = random.random()  % 4 - 1

best_loss = np.inf

for epochs in range(number_epochs):

	#Train the net on ONE sample (avant-garde)
	for i in range(samples):
		#Forward Propagation
		s1_t0 = 0.0
		s2_t0 = 0.0
		
		s1_t1 = sigmoid(b1 + s2_t0*w1 + X[i,0]*w_hat1)
		s2_t1 = sigmoid(b2 + s1_t0*w2 + X[i,1]*w_hat2)
		
		s1_t2 = sigmoid(b1 + s2_t1*w1)
		s2_t2 = sigmoid(b2 + s1_t1*w2)
		
		s1_t3 = sigmoid(b1 + s2_t2*w1)
		s2_t3 = sigmoid(b2 + s1_t2*w2)

		#Back propagation through time
		
		#Bias gradients
		delB1_3 = (Y[i,0] - s1_t3)*(1-s1_t3)*s1_t3
		delB2_3 = (Y[i,1] - s2_t3)*(1-s2_t3)*s2_t3
		
		delB1_2 = delB2_3 * w2 * (1 - s1_t2)*s1_t2
		delB2_2 = delB1_3 * w1 * (1 - s2_t2)*s2_t2
		
		delB1_1 = delB2_2 * w2 * (1 - s1_t1)*s1_t1
		delB2_1 = delB1_2 * w1 * (1 - s2_t1)*s2_t1
		#Update biases
		b1 = b1 + learning_rate * (delB1_1 + delB1_2 + delB1_3)
		b2 = b2 + learning_rate * (delB2_1 + delB2_2 + delB2_3)
		
		#State-weight gradients
		delW1_3 = (Y[i,0] - s1_t3)*(1-s1_t3)*s1_t3*s2_t2
		delW2_3 = (Y[i,1] - s2_t3)*(1-s2_t3)*s2_t3*s1_t2
		
		delW1_2 = delW2_3*w2*(1-s1_t2)*s2_t1
		delW2_2 = delW1_3*w1*(1-s2_t2)*s1_t1
		#Update state-weights
		w1 = w1 + learning_rate*(delW1_2 + delW1_3)
		w2 = w2 + learning_rate*(delW2_2 + delW2_3)
		
		#Input-weight gradients
		delW_hat1 = delB1_1*X[i,0]
		delW_hat2 = delB2_1*X[i,1]
		#Update input-weights
		w_hat1 = w_hat1 + learning_rate*delW_hat1
		w_hat2 = w_hat2 + learning_rate*delW_hat2

	#Predictions
	outputs = np.ones((samples,2))
	for i in range(samples):
		#Forward Propagation
		s1_t0 = 0.0
		s2_t0 = 0.0
		
		s1_t1 = sigmoid(b1 + s2_t0*w1 + X[i,0]*w_hat1)
		s2_t1 = sigmoid(b2 + s1_t0*w2 + X[i,1]*w_hat2)
		
		s1_t2 = sigmoid(b1 + s2_t1*w1)
		s2_t2 = sigmoid(b2 + s1_t1*w2)
		
		s1_t3 = sigmoid(b1 + s2_t2*w1)
		s2_t3 = sigmoid(b2 + s1_t2*w2)
		outputs[i,0] = s1_t3
		outputs[i,1] = s2_t3


	loss = np.sum(np.abs(Y - outputs))	
	if loss < best_loss:
		best_loss = loss
		best_b1 = b1
		best_w1 = w1
		best_w_hat1 = w_hat1
		best_b2 = b2
		best_w2 = w2
		best_w_hat2 = w_hat2
		print loss
		#if loss < 10:
			#print b1, w1, w_hat1
			#print b2, w2, w_hat2
"""	

	accuracy = float(number_correct) / float(samples) * 100
	accuracies.append(accuracy)
	iteration.append(epochs)

	if accuracy > best_accuracy:
		best_accuracy = accuracy
		best_iteration = epochs
		best_b1, best_w1, best_w_hat1 = (b1, w1, w_hat1)
		best_b2, best_w2, best_w_hat2 = (b2, w2, w_hat2)
"""
outputs = np.ones((samples,2))
for i in range(samples):
		#Forward Propagation
		s1_t0 = 0.0
		s2_t0 = 0.0
		
		s1_t1 = sigmoid(b1 + s2_t0*w1 + X[i,0]*w_hat1)
		s2_t1 = sigmoid(b2 + s1_t0*w2 + X[i,1]*w_hat2)
		
		s1_t2 = sigmoid(b1 + s2_t1*w1)
		s2_t2 = sigmoid(b2 + s1_t1*w2)
		
		s1_t3 = sigmoid(b1 + s2_t2*w1)
		s2_t3 = sigmoid(b2 + s1_t2*w2)
		outputs[i,0] = s1_t3
		outputs[i,1] = s2_t3


print Y[0:20,:]
print "\n"
print outputs[0:20,:]
print best_b1, best_w1, best_w_hat1
print best_b2, best_w2, best_w_hat2

		