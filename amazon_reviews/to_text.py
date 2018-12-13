import numpy as np

def read_docs(path):
  whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

  c = 0
  with open(path, 'r') as f:

    for line in f.readlines():

      line = line[:-17]
      line = ' '.join(line.split(':'))
      
      line = ' '.join(line.split('.'))
      
      line = ' '.join(line.split('_'))
      line = ''.join(filter(whitelist.__contains__, line))
      line = ' '.join(line.split())

      with open('processed_acl/kitchen_reviews/neg/negative'+str(c)+'.txt', 'w') as g:
        g.write(line)
      
      c += 1

    

if __name__ == "__main__":
  read_docs('processed_acl/kitchen/negative.txt')