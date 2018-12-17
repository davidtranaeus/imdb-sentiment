# imdb-sentiment: A project in DD2418 sprakt18 HT18-1 Language Engineering at KTH.

## Made by David Tranæus (davidtra@kth.se) and Karl Andrén (karlandr@kth.se)

## What is it?
imdb-sentiment is machine learning project in sentiment analysis. A selected classifier is trained on IMDb reviews and then tested on Amazon reviews. The purpose is to examine how well a model can classify reviews on a topic which it has not been exposed to during training.

## How to execute the program
1. Install dependecies: sklearn, numpy and nltk (nltk.corpus and nltk.stem) using pip.
2. Run the sentiment classification by specifying a classifier, test set and a optional data representation with the flags:
  - --classifier ['regression' or 'mlp'], regression is logistic regression and mlp is a neural network
  - --test ['books' or 'kitchen'], books are book reviews from the Amazon dataset and kitchen are kitchen accessory reviews from the Amazon dataset.
  - -b and -t will implement binary data representation or tf-idf representation for the bag of words vectors. Term frequency count representation is default.
- Example running with python3: python3 sentiment.py --classifier 'regression' --test 'books'
3. The script will print the 10-fold cross-validation accuaracies, confusion matrix and the classification accuracy on the selected test set.

## Dataset references

1. IMDb dataset: Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher (2011) \emph{Learning Word Vectors for Sentiment Analysis} available: http://www.aclweb.org/anthology/P11-1015

2. Amazon dataset: John Blitzer, Mark Dredze, Fernando Pereira (2007) \emph{Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification} available: http://www.cs.jhu.edu/~mdredze/publications/sentiment_acl07.pdf