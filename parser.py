import os
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

class Parser(object):
    def __init__(self, filenames, nr_of_classes = 0, doTest=False):
        self.sources = filenames
        self.vocab = {}
        self.nr_docs = len(self.sources)
        self.wordMatrix = np.zeros([self.nr_docs,120000])
        self.docnr = 0
        self.wordnr = 0
        #Blacklist for the 30 most common words in English
        self.blacklist = ["the","be","to","of","and","a","in","that","have","I",
        "it","for","not","on","with","he","as","you","do","at","this","but","his","by"
        ,"from","they","we","say","her","she"]
        self.nr_of_classes = nr_of_classes
        self.targets = np.zeros([self.nr_docs,2])
        self.label = 0
        self.doTest = doTest
        self.lemma = WordNetLemmatizer()
        
    def clean_line(self, line):
        """Clean and lemmatize words in the line"""
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        answer = ''.join(filter(whitelist.__contains__, line))
        answer = ' '.join(answer.split())
        ' '.join([self.lemma.lemmatize(word, pos='v') for word in answer.split()])

        return [answer]

    def text_gen(self):
        """Generate text from the documents"""
        for fname in self.sources:
            if fname == "NewClass":
                self.label += 1
            else:    
                self.targets[self.docnr] = [self.docnr,self.label] 
                self.docnr += 1
                with open(fname, encoding='utf8', errors='ignore') as f:
                    for line in f:
                        yield self.clean_line(line)

    def build_vocabulary(self):
        """Build vocabulary and document-term matrix"""
        wh_lemmas = set(wordnet.all_lemma_names())

        for i in self.text_gen():
            for k in i[0].split():
                k = k.lower()
                if k not in self.blacklist and k in wh_lemmas:
                    if k not in self.vocab:
                        if self.doTest:
                            continue
                        self.vocab[k] = self.wordnr
                        self.wordnr += 1
                        self.wordMatrix[self.docnr-1,self.wordnr] += 1 
                    else:
                        self.wordMatrix[self.docnr-1,self.vocab.get(k)] += 1
        if self.doTest:
            self.remove_excess_elements_test()
        else:
            self.remove_excess_elements()

    def vocabulary_size(self):
        return len(self.vocab)

    def remove_excess_elements(self):
        self.wordMatrix = self.wordMatrix[:,:self.wordnr]

    def remove_excess_elements_test(self):
        self.wordMatrix = self.wordMatrix[:,:len(self.vocab)]

    def set_vocab(self, vocab):
        self.vocab = vocab

def train_data(dir_names = ["imdb_reviews/neg","imdb_reviews/pos"], nr_of_classes = 2, crop=False, sample_Size = None):
    """Returns the document matrix for the training and validation data"""
    filenames = []
    if not crop:
        for i in range(nr_of_classes):
            filenames = np.append(filenames, [os.path.join(dir_names[i], fn) for fn in os.listdir(dir_names[i])])
            filenames = np.append(filenames, ["NewClass"])
    else:
        for i in range(nr_of_classes):
            for fn in range(sample_Size):
                filenames = np.append(filenames, [os.path.join(dir_names[i], os.listdir(dir_names[i])[fn])])
            filenames = np.append(filenames, ["NewClass"])

    parser = Parser(filenames, nr_of_classes=2)
    parser.build_vocabulary()
    return parser.wordMatrix, parser.targets, parser.vocab

def test_data(vocab, dir_names, nr_of_classes = 2, test=True, crop=False, sample_Size=None):
    """Returns the document matrix for the test data. Uses the vocabulary that was used for creating the training matrix"""
    filenames = []
    if not crop:
        for i in range(nr_of_classes):
            filenames = np.append(filenames, [os.path.join(dir_names[i], fn) for fn in os.listdir(dir_names[i])])
            filenames = np.append(filenames, ["NewClass"])
    else:
        for i in range(nr_of_classes):
            for fn in range(sample_Size):
                filenames = np.append(filenames, [os.path.join(dir_names[i], os.listdir(dir_names[i])[fn])])
            filenames = np.append(filenames, ["NewClass"])

    parser = Parser(filenames, nr_of_classes=2 ,doTest=test)
    if not test:
        parser.build_vocabulary()
        return parser.wordMatrix, parser.targets
    else:
        parser.set_vocab(vocab)
        parser.build_vocabulary()
        return parser.wordMatrix, parser.targets