import numpy as np
import math as mt
class LogisticRegression:
	def __init__(self,updateRate):
		self.weights = np.random.rand(1*12)*10
		self.updateRate = updateRate
	def fit(self,inputData,resultData):
		def featureSum(weights,inputData,resultData):
			def alpha(result):
				return 1/1+mt.exp(-result)
			zeros = np.zeros(inputData.shape[1])
			for i in range(len(resultData)):
				zeros = zeros + inputData[i]*(resultData - alpha(inputData[i].dot(weights.T)))
			return zeros
		oldWeights =  self.weights
		newWeights = oldWeights + updateRate * featureSum(oldWeights,inputData,resultData)
		while not np.array_equals(oldWeights,newWeights):
			oldWeights = newWeights
			newWeights = oldWeights + updateRate * featureSum(oldWeights,inputData,resultData)
		self.weights = newWeights
	def predict(self,inputData,trueLabel):
		def alpha(result):
			return 1/1+mt.exp(-result)
		output = 1 if (alpha(inputData.dot(self.weights.T)) > 0.5) else 0
		return output == trueLabel

class LinearDiscriminantAnalysis:
	pass