import csv
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
## load the data from train and test csv file
comments = []
results = []
commentsTest = []
def filtWord(s):
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(s)
	filtered_sentence = ''
	for token in word_tokens:
		if token not in stop_words:
			filtered_sentence = filtered_sentence + token + ' '
	return filtered_sentence
with open('reddit_train.csv','r',encoding = 'utf-8',errors = 'ignore') as f:
	csv_reader = csv.reader(f,delimiter=',')
	for row in csv_reader:
		sentence = filtWord(row[1])
		comments.append(sentence)
		results.append(row[-1])
comments = comments[1:]
results = results[1:]
with open('reddit_test.csv','r',encoding = 'utf-8',errors = 'ignore') as f:
	csv_reader = csv.reader(f,delimiter = ',')
	for row in csv_reader:
		sentence = filtWord(row[1])
		commentsTest.append(sentence)
commentsTest = commentsTest[1:]

## vectorize the dataset

vectorizer = CountVectorizer()

vectors_train = vectorizer.fit_transform(comments)

vectors_test = vectorizer.transform(commentsTest)




