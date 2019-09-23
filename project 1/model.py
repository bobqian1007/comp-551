import numpy as np
import math as mt
class LogisticRegression(object):
	def __init__(self,updateRate):
		self.weights = np.random.rand(1,12)
		self.updateRate = updateRate
	def fit(self,inputData,resultData):
		def featureSum(weights,inputData,resultData):
			def alpha(result):
				return 1/(1+mt.exp(-result))    # 1 / ( 1+mt.exp(-result) ) ?
			zeros = np.zeros(inputData.shape[1])
			for i in range(len(resultData)):
				zeros = zeros + inputData[i]*(resultData[i] - alpha(inputData[i].dot(weights.T)))
			return zeros
		oldWeights =  self.weights
		newWeights = oldWeights + self.updateRate * featureSum(oldWeights,inputData,resultData)
		while np.amax(np.abs(np.subtract(oldWeights,newWeights))) > 0.1 :
			oldWeights = newWeights
			newWeights = oldWeights + self.updateRate * featureSum(oldWeights,inputData,resultData)
		self.weights = newWeights
	def predict(self,inputData,trueLabel):		# split multi test sets into a few 1*m sets?
		def alpha(result):
			return 1/(1+mt.exp(-result))
		output = 1 if (alpha(inputData.dot(self.weights.T)) > 0.5) else 0
		return output == trueLabel

class LinearDiscriminantAnalysis(object):
	
	def __init__(self):
		self.mean0, self.mean1 = 0,0
		self.n0, self.n1 = 0,0
		self.P0,self.P1 = 0,0
		self.cov_mat,self.inv_cov = 0,0

	def fit(self,inputData,resultData):		# this function takes complete data set. i.e. contain input & output in one set
		input = np.concatenate((inputData,np.array([resultData]).T),axis = 1)
		#print(input)
		set0 = input[input[:,-1] == 0][:,:-1]
		set1 = input[input[:,-1] == 1][:,:-1]

		self.n0 = len(set0)
		self.n1 = len(set1)

		'''
		if (n0+n1) != len(input):
			#raise exception
		'''

		self.P0 = self.n0 / (self.n0 + self.n1)
		self.P1 = self.n1 / (self.n0 + self.n1)

	
		self.mean0 = 1/self.n0 * np.sum(set0,axis=0)  #average all columns of set0 & set1
		self.mean1 = 1/self.n1 * np.sum(set1,axis=0)  #average all columns of set0 & set1
		
		#print(self.mean0)
		'''
		cov_mat = np.cov((input[:,:-1]).T)
		inv_cov = np.linalg.inv(cov_mat)
		'''
		def helper(input,mean):
			sum = 0
				#mean = comput_mean(input,output)
				#n0,n1,total = comput_num(input,output)
			total = self.n0 + self.n1
			for i in range(len(input)):
				x = input[i]			# x = the i-th row of input matrix
				sum = sum + (x-mean)*((x-mean).T)/(total-2)
			return sum
			
		self.cov_mat = self.cov_mat + helper(set0,self.mean0)
		self.cov_mat = self.cov_mat + helper(set1,self.mean1)
			
		self.inv_cov = 1/self.cov_mat

		'''
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
			'''


	def predict(self,input,output):
		'''
		n0,n1,total = fit.comput_num(input,output)
		mean0 = fit.comput_mean( fit.split(input,output,0) )
		mean1 = fit.comput_mean( fit.split(input,output,1) )
		sigma = fit.comput_sigma(input,output)
		x = input
		'''
		'''
		bound = (np.log(self.P1/self.P0) 
				-0.5 * np.dot(self.mean1, np.dot(inv_cov,mean1)) 
				+ 0.5*np.dot(mean0,np.dot(inv_cov,mean0)) 
				+ np.dot(input,np.dot(inv_cov,(mean0-mean1))) )
		'''
		bound = np.log(self.P1/self.P0) - 0.5 * np.dot((self.mean1).T, self.inv_cov * self.mean1) + 0.5*np.dot((self.mean0).T,self.inv_cov * self.mean0) + np.dot(input.T,self.inv_cov * (self.mean0-self.mean1)) 
				
		if bound > 0 :		# assume 2 represents y=0 branch. 4 represents y=1 branch
			result = 1
		else:
			result = 0
		

		return result==output
			



