import json
from collections import defaultdict
import math

from nltk.stem import PorterStemmer
ps = PorterStemmer()

input_file = "/root/testout/part-00000"

output_file = "inverted_index.json"

inverted_index = {}
word_frequencies = defaultdict(int)
document_lengths = defaultdict(int)

with open(input_file, "r") as f:
    for line in f:

        parts = line.strip().split("\t")

        if len(parts) != 2:
            continue  # Skip malformed lines

        word_id = parts[0]  # Word ID
        try:
            doc_list = eval(parts[1])  # Convert string to list of tuples
        except:
            continue  # Skip if the format is incorrect
        
        doc_map = {doc_id: count for doc_id, count in sorted(doc_list, key=lambda x: x[1], reverse=True)}
        inverted_index[word_id] = doc_map  # Store in dictionary

        word_frequencies[word_id] += sum(doc_map.values())

        for doc_id, count in doc_map.items():
            document_lengths[doc_id] += count

sorted_word_freqs = dict(sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True))
vocab = list(sorted_word_freqs.keys())[:200]

# query_words = ["olympic", "gold", "athens", "reuters", "stocks", "friday", "investment", "market", "prices", "draft", "selections", "changing", "oil", "prices", "senate", "passes", "new", "law", "cubs", "baseball", "stadium", "city", "chicago", "NFC", "north", "history"]
# vocab.extend([ps.stem(word) for word in query_words])

word_count = sum(word_frequencies.values())
M = len(document_lengths)

avdl = word_count / M

idf_matrix = {}

for vocab_word in inverted_index:
        k = len(inverted_index[vocab_word])
        idf_matrix[vocab_word] = math.log((M + 1) / k)

# Save inverted index as JSON
with open(output_file, "w") as f:
        json.dump({
        "inverted_index": inverted_index,
        "document_lengths": document_lengths,
        "vocab": vocab,
        "M": M,
        "avdl": avdl,
        "idf_matrix": idf_matrix
    }, f, indent=4)