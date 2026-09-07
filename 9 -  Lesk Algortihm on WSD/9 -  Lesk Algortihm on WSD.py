import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
# nltk.download('wordnet')
# nltk.download('punkt')

def simpleLesk(sentence, word):
    sentence_tokens = word_tokenize(sentence)
    maximumOverlap = 0
    bestSense = None
    for sense in wordnet.synsets(word):
        definition_tokens = word_tokenize(sense.definition())
        overlap = set(sentence_tokens).intersection(definition_tokens)
        for example in sense.examples():
            example_tokens = word_tokenize(example)
            overlap.update(set(sentence_tokens).intersection(example_tokens))
        if len(overlap) > maximumOverlap:
            bestSense = sense
            maximumOverlap = len(overlap)
    return bestSense

sentences = [
    "Time flies like an arrow; fruit flies like a banana.",
    "He saw a man on a hill with a telescope.",
    "I need to book a flight for my vacation."
]
ambiguousWords = ["flies", "saw", "book"]

for sentence, ambiguousWord in zip(sentences, ambiguousWords):
    lesk_sense = lesk(sentence, ambiguousWord)
    simpleLesk_sense = simpleLesk(sentence, ambiguousWord)

    print("Sentence:", sentence)
    print("Ambiguous Word:", ambiguousWord)
    print("Lesk Sense:")
    print("- Definition:", lesk_sense.definition())
    print("Simple Lesk Sense:")
    print("- Definition:", simpleLesk_sense.definition())
    print()
