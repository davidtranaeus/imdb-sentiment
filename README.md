# imdb-sentiment: A project in DD2418 sprakt18 HT18-1 Language Engineering at KTH
## How to execute the program
1. Install dependecies: sklearn, numpy and nltk (nltk.corpus and nltk.stem)
2. Run the sentiment classification by specifying a classifier, test set and a optional data representation with the flags:
  --classifier ['regression' or 'mlp'], regression is logistic regression and mlp is a neural network
  --test ['books' or 'kitchen'], books are book reviews from the Amazon dataset and kitchen are kitchen accessory reviews from the Amazon dataset.
  -b and -t will implement binary data representation or tf-idf representation for the bag of words vectors. Term frequency count representation is default.
3. The script will print the 10-fold cross-validation accuaracies, confusion matrix and the classification accuracy on the selected test set.
