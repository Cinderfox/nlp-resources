# # Using Spacy:

# import spacy
# nlp = spacy.load("en_core_web_sm")

# content = "Indian diplomats, who have dealt with Sri Lanka in the past, underlined that Delhi was able to get access to Wadge Bank and its rich resources. Former Indian High Commissioner to Sri Lanka, Ashok Kantha, said, “The agreement of June 1974 on the boundary in historic waters between India and Sri Lanka placed Katchatheevu on the Sri Lankan side but it also paved the way for a series of other pacts clarifying and confirming the maritime boundary with Sri Lanka, including the understanding of March 1976 which recognised India’s sovereignty over the Wadge Bank and its rich resources.”"
# doc = nlp(content)

# classes = {
#     "class_3": ["PERSON", "ORGANIZATION", "GPE"],
#     "class_4": ["PERSON", "ORGANIZATION", "GPE", "MISC"],
#     "class_7": ["PERSON", "NORP", "FACILITY", "ORGANIZATION", "GPE", "MONEY", "PRODUCT"]
# }

# def print_entities_with_labels_for_class(doc, entity_classes):
#     for ent in doc.ents:
#         if ent.label_ in entity_classes:
#             print(ent.label_, "\t\t", ent.text)
#     print()

# for class_name, entity_classes in classes.items():
#     print_entities_with_labels_for_class(doc, entity_classes)


# ================================================================================================================



# # Using NLTK:

# import nltk
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag, ne_chunk

# content = "Indian diplomats, who have dealt with Sri Lanka in the past, underlined that Delhi was able to get access to Wadge Bank and its rich resources. Former Indian High Commissioner to Sri Lanka, Ashok Kantha, said, “The agreement of June 1974 on the boundary in historic waters between India and Sri Lanka placed Katchatheevu on the Sri Lankan side but it also paved the way for a series of other pacts clarifying and confirming the maritime boundary with Sri Lanka, including the understanding of March 1976 which recognised India’s sovereignty over the Wadge Bank and its rich resources.”"
# tokens = word_tokenize(content)
# pos_tags = pos_tag(tokens)
# ne_tree = ne_chunk(pos_tags)

# def extract_entities(ne_tree, entity_classes, class_name):
#     print()
#     print(f"Entities for {class_name}:")
#     for subtree in ne_tree:
#         if isinstance(subtree, nltk.Tree):
#             label = subtree.label()
#             entity = " ".join([word for word, pos in subtree.leaves()])
#             if label in entity_classes:
#                 print(label, "\t\t", entity)

# def print_misc_entities(ne_tree, entity_classes):
#     for subtree in ne_tree:
#         if isinstance(subtree, nltk.Tree):
#             label = subtree.label()
#             entity = " ".join([word for word, pos in subtree.leaves()])
#             if label == "NE" and entity not in entity_classes:
#                 print("MISC", "\t\t", entity)
#     print()


# classes = {
#     "class_3": ["PERSON", "ORGANIZATION", "GPE"],
#     "class_4": ["PERSON", "ORGANIZATION", "GPE", "MISC"],
#     "class_7": ["PERSON", "NORP", "FACILITY", "ORGANIZATION", "GPE", "MONEY", "PRODUCT"]
# }
# for class_name, entity_classes in classes.items():
#     if class_name == "class_4":
#         extract_entities(ne_tree, entity_classes, class_name)
#         print_misc_entities(ne_tree, entity_classes)
#     else:
#         extract_entities(ne_tree, entity_classes, class_name)