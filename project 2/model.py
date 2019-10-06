import numpy as np
class naiveBayes(object):
	def __init__(self):
		self.classProbability = {}
		self.numOfClass = 0
		self.className = []
	def fit(OutputData,inputData):
		for i in OutputData:
			if i not in self.classProbability:
				self.classProbability[i] = 1
				self.className.append(i)
				self.numOfClass = self.numOfClass + 1
			else:
				self.classProbability[i] = classProbability[i] + 1
		for key in self.classProbability:
			self.classProbability[key] = self.classProbability[key]/len(OutputData)
		self.probTable = np.zeros((numOfClass,inputData.shape[1]))
		for i in range(len(OutputData)):
			for j in range(inputData.shape[1]):
				if inputData[i,j] != 0:
					self.probTable[self.className.index(OutputData[i]),j] = self.probTable[self.className.index(OutputData[i]),j] + 1
		self.probTable = self.probTable / (len(OutputData)/self.numOfClass)			
			