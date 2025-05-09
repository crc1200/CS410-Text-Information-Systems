import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import pandas as pd

import os
import requests
import re

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

from gensim.downloader import load

from sklearn.metrics.pairwise import cosine_similarity

# Load the Google News pretrained Word2Vec model (300-dimensional vectors)
model = load("word2vec-google-news-300")  

class TextRetrieval():

  #For preprocessing
  punctuations = ""
  stop_words=set()

  #For VSM definition
  vocab = np.zeros(200)
  dataset = None
  K = 3 #TODO try your own

  def __init__(self):
    ##
    #TODO: obtain the file "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    # and store it locally in a location accessible directly by this script (e.g. same directory don't use absolute paths)

    filename = "finalSet.csv"

    # if not os.path.exists(filename):
    #   url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    #   response = requests.get(url)
    #   with open(filename, "wb") as f:
    #       f.write(response.content)
    
    ### TODO: Initialize punctuations (a string) and stop_words (a set)

    self.punctuations = r'[^\w\s]'

    self.punctuations2 = '"\\,<>./?@#$%^&*_~/!()-[]{};:\''

    self.stop_words = set(stopwords.words('english'))

  def read_and_preprocess_Data_File(self):
    ### Reads the test.csv file and iterates over every document content (entry in the column 2)
    ### removes leading and trailing spaces, transforms to lower case, remove punctuation, and removes stopwords
    ### Stores the formated information in the same "dataset" object

    dataset = pd.read_csv("finalSet.csv",header=None)
    punctuations = self.punctuations
    stop_words = self.stop_words

    dataset.head()
    for index, row in dataset.iterrows():
      line = row[2].strip().lower()
    
      # CREDIT: Was assisted by ChatGPT to get specific r-strings

      # html tags
      line = remove_tags(line)
      # urls
      line = re.sub(r'https?://\S+', '', line)
      # punctuation
      line = re.sub(punctuations, '', line)
      line = line.translate(str.maketrans("", "", self.punctuations2))
      # numbers
      line = re.sub(r'\d+', '', line)

      words = []
      words_in_line = line.split()
      for word in words_in_line:
        to_examine = word.lower()
        if to_examine and to_examine not in stop_words:
          words.append(to_examine)

      dataset.loc[index, 2] = ' '.join(words)

    self.dataset = dataset #Set dataset as object attribute

  def build_vocabulary(self): #,collection):
    ### Return an array of 200 most frequent works in the collection
    ### dataset has to be read before calling the vocabulary construction

    word_counter = {}

    #TODO: Create a vocabulary. Assume self.dataset has been preprocessed. Count the ocurrance of the words in the dataset. Select the 200 most common words as your vocabulary vocab. 
    for line in self.dataset[2]:
      words_in_line = line.split(" ")

      for word in words_in_line:
          word_counter[word] = word_counter.get(word, 0) + 1

    sorted_dic = dict(sorted(word_counter.items(), key=lambda item: item[1], reverse=True))
    vocab = np.array(list(sorted_dic.keys())[:200])

    self.vocab = vocab

  def adapt_vocab_query(self,query):
    ### Updates the vocabulary to add the words in the query

    #TODO: Use self.vocab and check whether the words in query are included in the vocabulary
    #If a word is not present, add it to the vocabulary (new size of vocabulary = original + #of words not in the vocabulary)
    #you can use a local variable vocab to work your changes and then update self.vocab
    vocab = self.vocab

    punctuations = self.punctuations

    query_fixed = re.sub(r'\d+', '', query)

    query_fixed = query_fixed.lower().translate(str.maketrans("", "", punctuations))

    query_words = query.split(" ")

    for query_word in query_words:
      if query_word and query_word not in vocab:
        vocab = np.append(vocab, query_word)

    self.vocab = vocab

  #### Word2Vec

  def text2Word2Vec(self, text):

    words = text.split()
    word_vectors = []
    
    for word in words:
        if word in model.key_to_index:  # Check if word exists in Word2Vec model
            word_vectors.append(model.get_vector(word))
        else:
           word_vectors.append(np.zeros(300))

    return np.mean(word_vectors, axis=0)  # Average word vectors

  def word2Vec_score(self, query, doc):
     query_vector = self.text2Word2Vec(query)
     doc_vector = self.text2Word2Vec(doc)

     similarity = cosine_similarity(query_vector.reshape(1, -1), doc_vector.reshape(1, -1))[0, 0]

     return similarity

  def print_docs(self, query, relevence_docs):
    sorted_indices = np.argsort(relevance_docs)[::-1]  # Sort in descending order
 
    top_five = (sorted_indices[:5])
    bottom_five = (sorted_indices[-5:])

    print(f"<---------- Relevant Document Results for Query: {query} ---------->")
    print()
    c = 1
    for i in top_five:
       title = self.dataset[1][i]
       doc = self.dataset[2][i]
       print(f"Rank {c} Most Relevant Document ")
       print()
       print(title, doc)
       print()
       c += 1

    print ("<---------- End ---------->")
    print()
    print()

    # print(f"<---------- Irrelevant Document Results for Query: {query} ---------->")
    # print()
    # c = 1
    # for i in bottom_five:
    #    doc = self.dataset[2][i]
    #    print(f"Rank {c} Most Irrelevant Document ")
    #    print()
    #    print(doc)
    #    print()
    #    c += 1

    # print ("<---------- End ---------->")
    # print()
    # print()

  def generate_avdl(self):
     
     avdl = 0
     
     for d in self.dataset[2]:
        dl = d.split()
        avdl += len(dl)

     self.avdl = avdl/len(self.dataset[2])
  
  def execute_search_Word2Vec(self,query):
 
    relevances = np.zeros(self.dataset.shape[0]) #Initialize relevances of all documents to 0

    #TODO: Use self.vocab to compute the relevance/ranking score of each document in the dataset using word2VecScore
    for i, doc in enumerate(self.dataset[2]):
      score = self.word2Vec_score(query, doc)
      relevances[i] = score
    
    return relevances # in the same order of the documents in the dataset
  
  def compute_IDF(self, M, collection):
    ### M number of documents in the collection; collection: documents (i.e., column 3 (index 2) in the dataset)

    #To solve this question you should use self.vocab

    self.IDF  = np.zeros(self.vocab.size) #Initialize the IDFs to zero

    for i, word in enumerate(self.vocab):
        k = 1
        for doc in collection:
            doc_words = set(doc.split(" "))
            if word in doc_words:
                k += 1

        self.IDF[i] = np.log((M + 1) / k)

      

    #TODO: for word in vocab: Compute the IDF frequency of each word in the vocabulary

  def text2TFIDF(self,text, applyBM25_and_IDF=False):
    ### returns the bit vector representation of the text

    #TODO: Use self.vocab, self.K and self.IDF to compute the TF-IDF representation of the text

    vocab = self.vocab

    tfidfVector = np.zeros(vocab.size)

    K = 2
    b = 0.5

    for i, word in enumerate(vocab):
      score = 0
      if word in text.split():
        #TODO: Set the value of TF-IDF to be (temporarily) equal to the word count of word in the text
        score = text.split().count(word)
        if applyBM25_and_IDF:
            #TODO: update the value of the tfidfVector entry to be equal to BM-25 (of the word in the document) multiplied times the IDF of the word'
            dl_value_second = len(text.split()) / self.avdl
            dl_value = (1 - b + b * dl_value_second)

            score = ((K + 1) * score) / (score + (K * dl_value))
            score *= self.IDF[i]
        tfidfVector[i] = score
    return tfidfVector

  #grade (enter your code in this cell - DO NOT DELETE THIS LINE)
  def tfidf_score(self,query,doc, applyBM25_and_IDF=False):
    q = self.text2TFIDF(query)
    d = self.text2TFIDF(doc, applyBM25_and_IDF)

    #TODO: compute the relevance using q and d

    relevance = 0
    for i in range(q.size):
        relevance += q[i] * d[i]

    return relevance
  
  def execute_search_TF_IDF(self,query):
    #DIFF: Compute IDF
    self.adapt_vocab_query(query) #Ensure query is part of the "common language" of documents and query
    # global IDF
    self.compute_IDF(self.dataset.shape[0],self.dataset[2]) #IDF is needed for TF-IDF and can be precomputed for all words in the vocabulary and a given fixed collection (this excercise)

    #For this function, you can use self.IDF and self.dataset
    relevances = np.zeros(self.dataset.shape[0]) #Initialize relevances of all documents to 0

    #TODO: Use self.vocab to compute the relevance/ranking score of each document in the dataset using tfidf_score
    for i, doc in enumerate(self.dataset[2]):
      score = self.tfidf_score(query, doc, True)
      relevances[i] = score
    
    return relevances # in the same order of the documents in the dataset

if __name__ == '__main__':
    tr = TextRetrieval()
    tr.read_and_preprocess_Data_File() #builds the collection
    tr.build_vocabulary()#builds an initial vocabulary based on common words
    tr.generate_avdl()
    queries = ["young girl named alice finds a rabbit and goes to a wonderland"]
    # print("#########\n")
    # print("Results for Word2Vec")

    # for query in queries:
    #   print("QUERY:",query)
    #   relevance_docs = tr.execute_search_Word2Vec(query)
    #   tr.print_docs(query, relevance_docs)

    print("#########\n")
    print("Results for TF-IDF")
    for query in queries:
      print("QUERY:",query)
      relevance_docs = tr.execute_search_TF_IDF(query)
      tr.print_docs(query, relevance_docs)