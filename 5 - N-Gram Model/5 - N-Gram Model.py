import numpy as np
from scipy.sparse import dok_matrix
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from collections import Counter
from nltk.util import ngrams

with open("Lab 5 - N – Gram Model/input.txt", "r") as file:
    corpus = file.read()

    
corpus = re.sub(r'[^a-zA-Z0-9.!? ]', '', corpus)
sentences = sent_tokenize(corpus)
tokens = []
for sentence in sentences:
    words = word_tokenize(sentence)
    tokens.extend(words)

unigrams = tokens
bigrams = list(ngrams(tokens, 2))

unigram_counts = Counter(unigrams)
bigram_counts = Counter(bigrams)

unique_words = set(tokens)
word_to_index = {word: i for i, word in enumerate(unique_words)}

matrix_size = len(unique_words)
sparse_matrix = dok_matrix((matrix_size, matrix_size), dtype=np.int32)

V = len(unique_words) 
for (word1, word2), count in bigram_counts.items():
    i, j = word_to_index[word1], word_to_index[word2]
    sparse_matrix[i, j] = count + 1 

for i in range(matrix_size):
    for j in range(matrix_size):
        if sparse_matrix[i, j] == 0:
            sparse_matrix[i, j] = 1

print("Unigram Counts:")
for word, count in unigram_counts.items():
    print(f"{word}: {count}")

print("\nBigram Counts:")
for bigram, count in bigram_counts.items():
    print(f"{bigram[0]} {bigram[1]}: {count}")


def print_sparse_matrix(sparse_matrix, index_to_word):
    dense_matrix = sparse_matrix.todense()
    print("Sparse Matrix (as dense for visualization):")
    print("     ", end="")
    for i in range(len(index_to_word)):
        print(f"{index_to_word[i]:>7}", end=" ")
    print()
    for i, row in enumerate(dense_matrix):
        print(f"{index_to_word[i]:<5}", end=" ")
        for j in range(dense_matrix.shape[1]):
            val = row[0, j]
            print(f"{val:7d}", end=" ")
        print()


index_to_word = {index: word for word, index in word_to_index.items()}

print("\nSparse Matrix with Add-One Smoothing for Zero Counts:")
print_sparse_matrix(sparse_matrix, index_to_word)