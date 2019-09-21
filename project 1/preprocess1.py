import csv
import numpy as np
wines = []
with open('winequality-red.csv','r') as f:
	wines = list(csv.reader(f,delimiter=";"))
wines = np.array(wines[1:],dtype=np.float)

for i in range(wines.shape[0]):
	if wines[i,11] > 5:
		wines[i,11] = 1
	else:
		wines[i,11] = 0

breastCancerList = []
with open('breast-cancer-wisconsin.data','r') as f:
	for line in f.readlines():
		if not '?' in line:
			l = list(map(int,line.strip().split(',')[1:]))
			breastCancerList.append(l)

breastCancers = np.array(breastCancerList,dtype=int)
print(wines)
print(breastCancers)