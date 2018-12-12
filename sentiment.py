import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from parserr import *

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
  print('binary')
  return (samples > 0).astype(np.int)

def less_data(data, n_of_each = 5000):
  n_samples = data.shape[0]
  splt_point = int(n_samples/2)
  sample_dim = data.shape[1]
  
  less = np.zeros((n_of_each*2, sample_dim))
  less[:n_of_each] = data[:n_of_each]
  less[n_of_each:] = data[splt_point:splt_point+n_of_each]

  return less

if __name__ == "__main__":
  
  print('Reading training data')
  samples, targets_data, vocabulary = trainData()
  samples = samples[:-2]

  print('Extracting targets')
  targets = targets_data[:,1].reshape(-1,1)
  review_no = targets_data[:,0]

  print('Cropping training data')
  training_subset = 5000
  less_samples = less_data(samples, training_subset)
  less_targets = less_data(targets, training_subset)
  less_targets = np.ravel(less_targets)

  # X_train, X_val, y_train, y_val = train_test_split(
  #   less_samples, less_targets, train_size = 0.75)
  
  # for c in [0.01, 0.05, 0.25, 0.5, 1]: # regularization paramters
  #   lr = LogisticRegression(C=c) 
  #   lr.fit(X_train, y_train)
  #   score = lr.score(X_val, y_val)
  #   print(score)

  print('Training logistic regression')
  lr = LogisticRegression() 
  lr.fit(less_samples, less_targets)

  print('Reading test data')
  test_samples, test_targets_data = testData(vocab = vocabulary)
  print(test_samples.shape)
  test_samples = test_samples[:-2]
  n_test_samples = test_samples.shape[0]

  print('Extracting targets')
  test_targets = test_targets_data[:,1].reshape(-1,1)
  review_no = test_targets_data[:,0]

  print('Cropping test data')
  test_subset = 5000
  less_test_samples = less_data(test_samples, test_subset)
  less_test_targets = less_data(test_targets, test_subset)
  less_test_targets = np.ravel(less_test_targets)

  print('Accuracy: ', end='')
  print(lr.score(less_test_samples, less_test_targets))