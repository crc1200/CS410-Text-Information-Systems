# Code based on https://blog.devgenius.io/big-data-processing-with-hadoop-and-spark-in-python-on-colab-bff24d85782f
from operator import itemgetter
import sys
import ast

# input comes from STDIN
wordMap = {}
current_list = []
word = None
current_word = None
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    line = line.lower()

    # parse the input we got from mapper.py
    word, count_tuple = line.split('	', 1)
    word = word.strip().lower()
    
    try:
      docid, count = ast.literal_eval(count_tuple)

    except ValueError:
      #count was not a number, so silently
      #ignore/discard this line
      continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        current_list.append((docid, count))
        
    else:
        if current_word:
            # write result to STDOUT
            current_list.sort()
            print ('%s	%s' % (current_word, current_list))
        current_list = [(docid, count)]
        current_word = word
# do not forget to output the last word if needed!
if current_word == word:
    print( '%s	%s' % (current_word, current_list))
