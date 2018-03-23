#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 02:08:40 2017

@author: Mike
"""

from math import exp


def coefficients(data, learning_rate, epochs):
	coef = [0.0 for i in range(len(data[0]))]
	for epoch in range(epochs):
		sum_error = 0
		for row in data:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + learning_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + learning_rate * error * yhat * (1.0 - yhat) * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
	return coef

def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))


x = []
y = []
with open('file1.txt', 'r') as data_file:
        for line in data_file:
#Add a comment to this line
            split_line = line.split(',')
            x.append(split_line[0])
            y.append(split_line[1])
        x.pop(0)
        y.pop(0)
        for i in range(len(x)):
            x[i] = float(x[i])
            y[i] = int(y[i])

data = []
for i in range(len(x)):
    data.append([x[i],y[i]])

learning_rate = 0.4
epochs = 1000
    
betas = coefficients(data, learning_rate, epochs)

print(betas)
