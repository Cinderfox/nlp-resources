from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

# Function to apply corrections to stemmed words based on certain rules
def apply_correction(original_word, stemmed_word):
    if stemmed_word.endswith("ing"):
        return stemmed_word + "e"
    elif stemmed_word.endswith("ly"):
        return stemmed_word[:-2]
    # If the stemmed word ends with "ive" and the original also ends with "ive",
    # but the stemmed word does not end with "e", append "e" to it
    elif stemmed_word.endswith("ive") and original_word.endswith("ive") and not stemmed_word.endswith("e"):
        return stemmed_word + "e"
    
    return stemmed_word

input_file_path = "input.txt"
output_file_path = "output.txt"

with open(input_file_path, 'r') as input_file:
    words = [word.lower() for word in word_tokenize(input_file.read())]

corrected_roots = []

with open(output_file_path, 'w') as output_file:
    for w in words:
        stemmed_word = ps.stem(w)
        corrected_word = apply_correction(w, stemmed_word)
        corrected_roots.append(corrected_word)
        output_file.write("Original - {} \t\t Stemmed - {} \t\t Corrected - {}\n".format(w, stemmed_word, corrected_word))

print("Number of Corrected Roots:", len(corrected_roots))