import numpy as np
import math as mt
class LogisticRegression(object):
	def __init__(self,updateRate):
		self.weights = np.random.rand(1*12)*10
		self.updateRate = updateRate
	def fit(self,inputData,resultData):
		def featureSum(weights,inputData,resultData):
			def alpha(result):
				return 1/1+mt.exp(-result)    # 1 / ( 1+mt.exp(-result) ) ?
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
	def predict(self,inputData,trueLabel):		# split multi test sets into a few 1*m sets?
		def alpha(result):
			return 1/1+mt.exp(-result)
		output = 1 if (alpha(inputData.dot(self.weights.T)) > 0.5) else 0
		return output == trueLabel

class LinearDiscriminantAnalysis(object):
	
	def __init__(self)：
		pass

	def fit(self,input,output):

		def compute_num(input,output):  # count the number of set(y=0),set(y=1) and total input sets
			n0,n1 = 0,0
			for i in range(len(output)):			# psuedo code
				if output[i][0] == 0 :
					n0 = n0 + 1
				else
					n1 = n1 + 1
			total = no + n1
			return n0,n1,total						

		def split(input,output,y):		# this function is used to split input into two matrices according to y=0 or 1
			pass						# return newInput,newOutput

		def compute_mean(input,output):
			return 1/(len(input)) * np.sum(input,axis=0)  #average all columns of input matrix

		def compute_sigma(input,output)：

			sigma = 0

			def helper(input,output):
				sum = 0
				mean = comput_mean(input,output)
				n0,n1,total = comput_num(input,output)
				for i in range(len(input)):
					x = input[i]			# x = the i-th row of input matrix
					sum = sum + (x-mean)*((x-mean).T)/(total-2)

			sigma = sigma + helper(split(input,output,0))
			sigma = sigma + helper(split(input,output,1))

			return sigma


	def predict(self,input,output):

		n0,n1,total = fit.comput_num(input,output)
		mean0 = fit.comput_mean( fit.split(input,output,0) )
		mean1 = fit.comput_mean( fit.split(input,output,1) )
		sigma = fit.comput_sigma(input,output)
		x = input

		bound = mt.log((n1/n0),10) - 1/2*(mean1.T)*1/sigma*mean1 +1/2*(mean0.T)*1/sigma*mean0+(x.T)*1/sigma*(mean1-mean0)

		return bound>0		# if bound>0 then y=1 else y=0
			



