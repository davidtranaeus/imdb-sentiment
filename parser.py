import os
import argparse
import time
import string
import numpy as np


class Parser(object):
    def __init__(self, filenames, nrofclasses = 0):
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
        self.nrofclasses = nrofclasses
        self.targets = np.zeros([self.nr_docs-self.nrofclasses,2])
        self.label = 0
        

    def clean_line(self, line):

        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        # Remove characters in line that is not whitelisted and join together
        answer = ''.join(filter(whitelist.__contains__, line))
        answer = ' '.join(answer.split())
        # Return clean a line
        return [answer]


    def text_gen(self):
        # Cleans and returns all lines 
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
     
        # Iterate through cleaned lines and add word to vocab if it doesn't exist
        # Also build a word vector for each document
        for i in self.text_gen():
            for k in i[0].split():
                k = k.lower()
                # print(k)
                if k not in self.blacklist:
                    if k not in self.vocab:
                        self.vocab[k] = self.wordnr
                        self.wordnr += 1
                        # print(self.docnr)
                        self.wordMatrix[self.docnr-1,self.wordnr] += 1 
                    else:
                        self.wordMatrix[self.docnr-1,self.vocab.get(k)] += 1

        self.remove_excess_elements()
        # self.remove_excess_elements_targets()





    def vocabulary_size(self):
        return len(self.vocab)

    def remove_excess_elements(self):
        self.wordMatrix = self.wordMatrix[:,1:self.wordnr]

    # def remove_excess_elements_targets(self):
    #     self.targets = self.targets[:self.nr_docs-self.nrofclasses,:]


    
def trainData(dir_names = ["aclImdb/train/neg","aclImdb/train/pos"], nrofclasses = 2):
    filenames = []
    for i in range(nrofclasses):
        filenames = np.append(filenames, [os.path.join(dir_names[i], fn) for fn in os.listdir(dir_names[i])])
        filenames = np.append(filenames, ["NewClass"])

    parser = Parser(filenames,nrofclasses=2)
    parser.build_vocabulary()
    return parser.wordMatrix, parser.targets

def testData(dir_names = ["aclImdb/test/neg","aclImdb/test/pos"], nrofclasses = 2):
    filenames = []
    for i in range(nrofclasses):
        filenames = np.append(filenames, [os.path.join(dir_names[i], fn) for fn in os.listdir(dir_names[i])])
        filenames = np.append(filenames, ["NewClass"])

    parser = Parser(filenames,nrofclasses=2)
    parser.build_vocabulary()
    return parser.wordMatrix, parser.targets


if __name__ == '_main_':
    # dir_name = ["aclImdb/train/neg","aclImdb/train/pos"]
    dir_name = ["aclImdb/test/neg","aclImdb/train/pos"]

    filenames = []
    nrofclasses = 2
    for i in range(nrofclasses):
        filenames = np.append(filenames, [os.path.join(dir_name[i], fn) for fn in os.listdir(dir_name[i])])
        filenames = np.append(filenames, ["NewClass"])

    parser = Parser(filenames)
    parser.build_vocabulary()
    # print(parser.vocab)
    # print(parser.wordnr)
    # print(parser.wordMatrix.shape)
    # print(parser.wordMatrix[:2,:200])
    # print(parser.targets[:,:])