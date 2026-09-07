from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('Lab 3 - Cosine Similarity/input-1.txt', 'r') as file:
    text1 = file.read()

with open('Lab 3 - Cosine Similarity/input-2.txt', 'r') as file:
    text2 = file.read()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
tokens1 = [ps.stem(word) for word in word_tokenize(text1.lower()) if word.lower() not in stop_words]
tokens2 = [ps.stem(word) for word in word_tokenize(text2.lower()) if word.lower() not in stop_words]

tokens3 = [ps.stem(word) for word in word_tokenize(text1.lower())]
tokens4 = [ps.stem(word) for word in word_tokenize(text2.lower())]

tokens5 = [word for word in word_tokenize(text1.lower())]
tokens6 = [word for word in word_tokenize(text2.lower())]

X = CountVectorizer().fit_transform([' '.join(tokens1), ' '.join(tokens2)])
Y = CountVectorizer().fit_transform([' '.join(tokens3), ' '.join(tokens4)])
Z = CountVectorizer().fit_transform([' '.join(tokens5), ' '.join(tokens6)])
dense_matrix = X.toarray()

cosine_sim = cosine_similarity(X)
cosine_sim2 = cosine_similarity(Y)
cosine_sim3 = cosine_similarity(Z)

print()
print()
print(f"Cosine Similarity without stopwords: \t\t\t {cosine_sim[0][1]}")
print(f"Cosine Similarity with stopwords: \t\t\t {cosine_sim2[0][1]}")
print(f"Cosine Similarity with stopwords & without portstemmer:  {cosine_sim3[0][1]}")
print()
print()