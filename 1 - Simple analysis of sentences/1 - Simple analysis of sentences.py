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

# Tokenize the text into words including punctuation
tokens_with_punctuation = word_tokenize(text)

# Create a tokenizer to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')
# Tokenize the text into words without punctuation
tokens_without_punctuation = tokenizer.tokenize(text)

# Define a set of stop words in English
stop_words = set(stopwords.words('english'))
# Filter out stop words from the tokens
tokens_filtered = [word for word in tokens_without_punctuation if word.lower() not in stop_words]

# Generate part-of-speech tags for the tokens
pos_tags = pos_tag(tokens_with_punctuation)

# Map POS tags to simplified labels
pos_label_map = {
    'N': 'Noun',
    'V': 'Verb',
    'R': 'Adverb',
    'J': 'Adjective',
    'D': 'Determiner',
    'P': 'Preposition',
    'C': 'Conjunction',
}

# Function to simplify part-of-speech tags
def simplify_pos_tag(tag):
    return pos_label_map.get(tag[0], 'Other')

# Simplify the POS tags for the tokens
simplified_pos_tags = [simplify_pos_tag(tag) for word, tag in pos_tags]

# Write results to the output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # Output unique tokens including punctuation
    print("\nUnique Tokens Including Punctuation:", file=output_file)
    print(set(tokens_with_punctuation), file=output_file)

    # Output unique tokens excluding punctuation and stopwords
    print("\nUnique Tokens Excluding Punctuation and Stopwords:", file=output_file)
    print(set(tokens_filtered), file=output_file)

    # Function to analyze tokens and write statistics to the output file
    def analyze_tokens(tokens, title):
        print(f"\nAnalysis for {title}:\n", file=output_file)

        # Calculate frequency distribution of tokens
        freq_dist = FreqDist(tokens)
        print(f"Most common words: {freq_dist.most_common()}\n", file=output_file)

        # Count occurrences of simplified POS tags
        pos_counts = Counter(simplified_pos_tags)
        print(f"Major Parts of Speech: {pos_counts}\n", file=output_file)

        # Calculate average word length
        avg_word_length = sum(len(word) for word in tokens) / len(tokens)
        print(f"Average word length: {avg_word_length:.2f}\n", file=output_file)

        # Count total number of words
        total_words = len(tokens)
        print(f"Total number of words: {total_words}\n", file=output_file)

    # Analyze and output results for tokens with punctuation
    analyze_tokens(tokens_with_punctuation, "Tokens Including Punctuation")
    # Analyze and output results for filtered tokens
    analyze_tokens(tokens_filtered, "Tokens Excluding Punctuation and Stopwords")
