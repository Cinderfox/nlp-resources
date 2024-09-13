import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter

file_path = 'input.txt'
output_file_path = 'output.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

tokens_with_punctuation = word_tokenize(text)

tokenizer = RegexpTokenizer(r'\w+')
tokens_without_punctuation = tokenizer.tokenize(text)

stop_words = set(stopwords.words('english'))
tokens_filtered = [word for word in tokens_without_punctuation if word.lower() not in stop_words]

pos_tags = pos_tag(tokens_with_punctuation)

pos_label_map = {
    'N': 'Noun',
    'V': 'Verb',
    'R': 'Adverb',
    'J': 'Adjective',
    'D': 'Determiner',
    'P': 'Preposition',
    'C': 'Conjunction',
}


def simplify_pos_tag(tag):
    return pos_label_map.get(tag[0], 'Other')


simplified_pos_tags = [simplify_pos_tag(tag) for word, tag in pos_tags]

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    print("\nUnique Tokens Including Punctuation:", file=output_file)
    print(set(tokens_with_punctuation), file=output_file)

    print("\nUnique Tokens Excluding Punctuation and Stopwords:", file=output_file)
    print(set(tokens_filtered), file=output_file)

    def analyze_tokens(tokens, title):
        print(f"\nAnalysis for {title}:\n", file=output_file)

        freq_dist = FreqDist(tokens)
        print(f"Most common words: {freq_dist.most_common()}\n", file=output_file)

        pos_counts = Counter(simplified_pos_tags)
        print(f"Major Parts of Speech: {pos_counts}\n", file=output_file)

        avg_word_length = sum(len(word) for word in tokens) / len(tokens)
        print(f"Average word length: {avg_word_length:.2f}\n", file=output_file)

        total_words = len(tokens)
        print(f"Total number of words: {total_words}\n", file=output_file)

    analyze_tokens(tokens_with_punctuation, "Tokens Including Punctuation")
    analyze_tokens(tokens_filtered, "Tokens Excluding Punctuation and Stopwords")