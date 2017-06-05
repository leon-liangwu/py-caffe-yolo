import math
import numpy as np


def logistic_activate(x):
	return 1./(1. + np.exp(-x))

def logistic_gradient(x):
	return (1-x)*x

def softmax(input):
	sum_squre = np.sum(input**2)
	output = input / sum_squre
	return output
