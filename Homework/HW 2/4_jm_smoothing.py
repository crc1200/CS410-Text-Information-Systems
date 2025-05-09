import json
import pandas as pd
import math

from collections import Counter
from collections import defaultdict
from nltk.stem import PorterStemmer
ps = PorterStemmer()

vocab = None
inverted_index = None
dataset = None
document_lengths = None

lambda_val = 0.1

default_prob_value = 1e-10

smoothing_value = (1 - lambda_val) / lambda_val

K = 3
b = 0.5

with open("inverted_index.json", "r") as f:
    data = json.load(f)

inverted_index = data["inverted_index"]
document_lengths = data["document_lengths"]
vocab = data["vocab"]

word_count = sum(document_lengths.values())  # Total words in collection

collection_probs = {
    word: sum(docs.values()) / word_count
    for word, docs in inverted_index.items()
}

def execute_search(word_counts):
    
    accumulator = {}

    for base_word, query_count in word_counts.items():
            query_word = ps.stem(base_word)
            if query_word in inverted_index:
                
                documents = inverted_index[query_word]
                
                for doc_id, doc_count in documents.items():

                    doc_length = document_lengths[doc_id]

                    denominator = doc_length * collection_probs.get(query_word, default_prob_value)
                    
                    right_value = 1 + smoothing_value * (doc_count / denominator)

                    score = query_count * math.log(right_value)

                    accumulator[doc_id] = accumulator.get(doc_id, 0) + score

    return accumulator

def determine_query_vectors(queries):
    query_vectors = {} 

    for query in queries:
        words = query.split() 
        query_vectors[query] = Counter(words)

    return query_vectors

def print_docs(query, relevence_docs):
    sorted_word_freqs = dict(sorted(relevence_docs.items(), key=lambda item: item[1], reverse=True))
    top_five = list(sorted_word_freqs.keys())[:5]
    bottom_five = list(sorted_word_freqs.keys())[-5:][::-1]

    print(f"<---------- Relevant Document Results for Query: {query} ---------->")
    print()
    c = 1
    for ind in top_five:
       i = int(ind) - 1
       score = sorted_word_freqs[ind]
       title = dataset[1][i]
       doc = dataset[2][i]
       print(f"Rank {c} Most Relevant Document ")
       print()
    #    print("Score: ", score, "Title:", title, "Document: ", doc)
       print(f"Score: {score}, Title: {title}, Document: {doc}")
       print()
       c += 1

    print ("<---------- End ---------->")
    print()
    print()

    # prints out the lowest five documents that have at least a query word in the document (Was asked in Campus Wire Post #224)
    print(f"<---------- Irrelevant Document Results for Query: {query} ---------->")
    
    print()
    c = 1
    for ind in bottom_five:
       i = int(ind) - 1
       score = sorted_word_freqs[ind]
       title = dataset[1][i]
       doc = dataset[2][i]
       print(f"Rank {c} Most Irrelevant Document ")
       print()
       print(score, title, doc)
       print()
       c += 1

    print ("<---------- End ---------->")
    print()
    print()

if __name__ == '__main__':
   dataset = pd.read_csv("train.csv",header=None)

#    queries = ["olympic gold athens", "reuters stocks friday", "investment market prices"]
   queries = ["nfl draft selections", "changing oil prices", "senate passes new law"]


   query_vectors = determine_query_vectors(queries)

   for query, word_counts in query_vectors.items():
        print("QUERY: ", query)
        relevence_docs = execute_search(word_counts)
        print_docs(query, relevence_docs)

