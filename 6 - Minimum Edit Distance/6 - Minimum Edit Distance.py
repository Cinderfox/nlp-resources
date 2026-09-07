import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return filtered_words

def bigram_similarity(str1, str2):
    bigrams1 = set(nltk.bigrams(str1))
    bigrams2 = set(nltk.bigrams(str2))

    intersection = bigrams1.intersection(bigrams2)
    union = bigrams1.union(bigrams2)

    similarity = len(intersection) / len(union)
    return similarity

def bigram_edit_distance(str1, str2):
    bigrams1 = list(nltk.bigrams(str1))
    bigrams2 = list(nltk.bigrams(str2))

    distance = nltk.edit_distance(bigrams1, bigrams2)

    edit_operations = extract_edit_operations(bigrams1, bigrams2)

    return distance, edit_operations

def find_word_with_least_edit_distance(user_input, word_list):
    min_distance = float('inf')
    closest_words = []

    for word in word_list:
        distance, _ = bigram_edit_distance(user_input, word)
        if distance < min_distance:
            min_distance = distance
            closest_words = [word]
        elif distance == min_distance:
            closest_words.append(word)

    return closest_words, min_distance

def count_matching_bigrams(str1, str2):
    bigrams1 = set(nltk.bigrams(str1))
    bigrams2 = set(nltk.bigrams(str2))

    matching_bigrams = bigrams1.intersection(bigrams2)
    return len(matching_bigrams)

def extract_edit_operations(str1, str2):
    table = [[None] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    for i in range(len(str1) + 1):
        for j in range(len(str2) + 1):
            if i == 0:
                table[i][j] = ('I', j)
            elif j == 0:
                table[i][j] = ('D', i)
            else:
                insertion_cost = table[i][j - 1][1] + 1
                deletion_cost = table[i - 1][j][1] + 1
                substitution_cost = table[i - 1][j - 1][1] + int(str1[i - 1] != str2[j - 1])

                min_cost = min(insertion_cost, deletion_cost, substitution_cost)

                if min_cost == insertion_cost:
                    table[i][j] = ('I', j)
                elif min_cost == deletion_cost:
                    table[i][j] = ('D', i)
                else:
                    table[i][j] = ('S' if str1[i - 1] != str2[j - 1] else 'M', substitution_cost)

    i, j = len(str1), len(str2)
    edit_operations = []

    while i > 0 or j > 0:
        op, cost = table[i][j]
        edit_operations.append((op, i - 1, j - 1))
        if op == 'I':
            j -= 1
        elif op == 'D':
            i -= 1
        else:
            i -= 1
            j -= 1

    return list(reversed(edit_operations))

def find_minimum_edit_distance_for_top_n(user_input, word_list, top_n=10):
    similarities = []

    for word in word_list:
        similarity = bigram_similarity(user_input, word)
        similarities.append((word, similarity))

    sorted_words = sorted(similarities, key=lambda x: x[1], reverse=True)

    print(f"{top_n} words similar to '{user_input}':")
    for word, sim in sorted_words[:top_n]:
        print(f"   {word}")

    print("\nCalculating minimum edit distance for the 10 similar words:")
    for word, _ in sorted_words[:top_n]:
        distance, edit_operations = bigram_edit_distance(user_input, word)
        print(f"   {word}: {distance} edit(s)")
        if edit_operations:
            # print(f"      Edit Operations: {edit_operations}")

            input_bigrams = list(nltk.bigrams(user_input))
            selected_word_bigrams = list(nltk.bigrams(word))

            print(f"      Bigrams for input word: {input_bigrams}")
            print(f"      Bigrams for selected word '{word}': {selected_word_bigrams}")

            matching_bigrams_count = count_matching_bigrams(input_bigrams, selected_word_bigrams)
            print(f"      Matching Bigrams Count: {matching_bigrams_count + 1}")

            print(f"      Total number of bigrams for input word: {len(input_bigrams)}")
            print(f"      Total number of bigrams for selected word '{word}': {len(selected_word_bigrams)}")

if __name__ == "__main__":
    with open('Lab 6 - Minimum Edit Distance/input.txt', 'r') as file:
        paragraph = file.read()
    word_list = preprocess_text(paragraph)
    user_input = input("Enter a word: ")

    similarities = []

    for word in word_list:
        similarity = bigram_similarity(user_input, word)
        similarities.append((word, similarity))

    sorted_words = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

    print(f"\nTop 10 words similar to '{user_input}':")
    for word, sim in sorted_words:
        print(f"   {word}")

    print("\nCalculating minimum edit distance for the top 10 similar words:")
    for word, _ in sorted_words:
        distance, edit_operations = bigram_edit_distance(user_input, word)
        print(f"   {word}: {distance} edit(s)")
        if edit_operations:
            # print(f"      Edit Operations: {edit_operations}")

            input_bigrams = list(nltk.bigrams(user_input))
            selected_word_bigrams = list(nltk.bigrams(word))

            print(f"      Bigrams for input word: {input_bigrams}")
            print(f"      Bigrams for selected word '{word}': {selected_word_bigrams}")

            matching_bigrams_count = count_matching_bigrams(input_bigrams, selected_word_bigrams)
            print(f"      Matching Bigrams Count: {matching_bigrams_count + 1}")

            print(f"      Total number of bigrams for input word: {len(input_bigrams)}")
            print(f"      Total number of bigrams for selected word '{word}': {len(selected_word_bigrams)}")

    closest_words, min_distance = find_word_with_least_edit_distance(user_input, [word for word, _ in sorted_words])
    print(f"\n\nAll words with the least edit distance to '{user_input}' is {closest_words} with {min_distance} edit(s).")