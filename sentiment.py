import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from parser import *

def tf_idf(samples):
  """tf-idf representation"""
  n_cols = samples.shape[1]
  n_samples = samples.shape[0]

  df = dict()
  for i in range(n_cols):
    df[i] = len(np.nonzero(samples[:,i]))

  for i in range(n_samples):
    for j in range(n_cols):
      samples[i,j] *= (n_samples/df[j])

  return samples

def binary(samples):
  """one hot encoding representation"""
  return (samples > 0).astype(np.int)

if __name__ == "__main__":
  
  # train_samples, train_targets_data = trainData()

  # targets = targets_data[:,1]
  # review_no = targets_data[:,0]

  # X_train, X_val, y_train, y_val = train_test_split(
  #   samples, targets, train_size = 0.75)

  # for c in [0.01, 0.05, 0.25, 0.5, 1]: # regularization paramters

  #   lr = LogisticRegression(c=c) 
  #   lr.fit(X_train, y_train)
  #   score = lr.score(X_val, y_val)
  #   print(score)



  # Vector extraction
  # reviews = [
  #   'This is the first document.',
  #   'This document is the second document.',
  #   'And this is the third one.',
  #   'Is this the first document?',
  #   ]

  # cv = CountVectorizer()
  # x = cv.fit_transform(reviews)
  # print(x.toarray())