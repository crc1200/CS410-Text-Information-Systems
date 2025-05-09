from collections import Counter
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# description = "Four Weddings and a Funeral, Document: Four Weddings And A Funeral is a British comedy about a British Man named Charles and an American Woman named Carrie who go through numerous weddings before they determine if they are right for one another."
# description = "In the sequel to Tim Burton's "Alice in Wonderland", Alice Kingsleigh returns to Underland and faces a new adventure in saving the Mad Hatter."


words = description.split()

for s in words:

    print(ps.stem(s))