import nltk
import time
cfg = nltk.CFG.fromstring("""
    S -> VP | VP PP
    VP -> V | V NP | VP NP | VP Adv
    NP -> Det N | Adj NP | N | NP PP
    PP -> P NP
    Adv -> 'quickly' | 'happily' | 'carefully' | 'while'
    Det -> 'the' | 'a'
    N -> 'man' | 'dog' | 'park' | 'food' | 'working'
    Adj -> 'good' | 'happy' | 'big' | 'colorful'
    V -> 'Eat' | 'be' | 'run' | 'read' | 'ran'
    P -> 'to' | 'during' | 'with' | 'on'
""")


parser = nltk.ChartParser(cfg)

# sentence = "run carefully"
sentence = "Eat good food during working"

for i, tree in enumerate(parser.parse(sentence.split()), 1):
    start_time = time.time()
    tree.pretty_print()
    end_time = time.time()
    print(f"Time taken to parse tree {i}: {end_time - start_time:.4f} seconds")