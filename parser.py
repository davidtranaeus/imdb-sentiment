import os
import argparse
import time
import string
import numpy as np


class Parser(object):
    def __init__(self, filenames):
        self.sources = filenames
        self.vocab = {}
        self.nr_docs = len(self.sources)
        self.wordMatrix = np.zeros([self.nr_docs,50000])
        self.docnr = 0
        self.wordnr = 0
        #Blacklist for the 30 most common words in English
        self.blacklist = ["the","be","to","of","and","a","in","that","have","I",
        "it","for","not","on","with","he","as","you","do","at","this","but","his","by"
        ,"from","they","we","say","her","she"]
        
        

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




    def vocabulary_size(self):
        return len(self.vocab)

    def remove_excess_elements(self):
        self.wordMatrix = self.wordMatrix[:,1:self.wordnr]

    
        

if __name__ == '__main__':
    # dir_name = "corpus"
    dir_name = ["corpus/architecture","corpus/bio","corpus/cs","corpus/indek","corpus/sciences"]
    filenames = []
    for i in range(5):
        filenames = np.append(filenames, [os.path.join(dir_name[i], fn) for fn in os.listdir(dir_name[i])])

    parser = Parser(filenames)
    parser.build_vocabulary()
    # print(parser.vocab)
    # print(parser.wordnr)
    # print(parser.wordMatrix.shape)
    # print(parser.wordMatrix[:2,:200])

