import sys
import io
import re
import csv
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='latin1')
reader = csv.reader(input_stream)

docid = 1  # Assuming 1-indexed for now

for row in reader:
    if len(row) < 3:
        continue  # skip malformed rows
    description = row[2]  # 3rd column is description

    description = re.sub(r'[^a-zA-Z\s]', '', description)
    description = description.lower()
    words = description.split()

    wordMap = {}

    for word in words:
        if word not in stop_words:
            stemmed_word = ps.stem(word)
            wordMap[stemmed_word] = wordMap.get(stemmed_word, 0) + 1

    for word, value in wordMap.items():
        print('%s\t%s' % (word, (docid, value)))

    docid += 1