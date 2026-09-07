from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
import string

# nltk.download('punkt')
# nltk.download('stopwords')

text = """A great transformation can be observed in our daily routine life along with the increasing involvement of IoT devices and technology. One such development of IoT is the concept of Smart Home Systems (SHS) and appliances that consist of internet based devices, automation system for homes and reliable energy management system [3]. Besides, another important achievement of IoT is Smart Health Sensing system (SHSS). SHSS incorporates small intelligent equipment and devices to support the health of the human being. These devices can be used both indoors and outdoors to check and monitor the different health issues and fitness level or the amount of calories burned in the fitness center etc. Also, it is being used to monitor the critical health conditions in the hospitals and trauma centers as well. Hence, it has changed the entire scenario of the medical domain by facilitating it with high technology and smart devices [4, 5]. Moreover, IoT developers and researchers are actively involved to uplift the life style of the disabled and senior age group people. IoT has shown a drastic performance in this area and has provided a new direction for the normal life of such people. As these devices and equipment are very cost effective in terms of development cost and easily available within a normal price range, hence most of the people are availing them [6]. Thanks to IoT, as they can live a normal life. Another important aspect of our life is transportation. IoT has brought up some new advancements to make it more efficient, comfortable and reliable. Intelligent sensors, drone devices are now controlling the traffic at different signalized intersections across major cities. In addition, vehicles are being launched in markets with pre-installed sensing devices that are able to sense the upcoming heavy traffic congestions on the map and may suggest you another route with low traffic congestion [7]. Therefore IoT has a lot to serve in various aspects of life and technology. We may conclude that IoT has a lot of scope both in terms of technology enhancement and facilitate the humankind.

IoT has also shown its importance and potential in the economic and industrial growth of a developing region. Also, in trade and stock exchange market, it is being considered as a revolutionary step. However, security of data and information is an important concern and highly desirable, which is a major challenging issue to deal with [5]. Internet being a largest source of security threats and cyber-attacks has opened the various doors for hackers and thus made the data and information insecure. However, IoT is committed to provide the best possible solutions to deal with security issues of data and information. Hence, the most important concern of IoT in trade and economy is security. Therefore, the development of a secure path for collaboration between social networks and privacy concerns is a hot topic in IoT and IoT developers are working hard for this.

The remaining part of the article is organized as follows: “Literature survey” section will provide state of art on important studies that addressed various challenges and issues in IoT. “IoT architecture and technologies” section discussed the IoT functional blocks, architecture in detail. In “Major key issues and challenges of IoT” section, important key issues and challenges of IoT is discussed. “Major IoT applications” section provides emerging application domains of IoT. In “Importance of big data analytics in IoT” section, the role and importance of big data and its analysis is discussed. Finally, the article concluded in “Conclusions” section."""



text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation.replace('.', '')))
sentences = sent_tokenize(text_no_punctuation)

word_freq_per_sentence = {}

for idx, sentence in enumerate(sentences, 1):
    unique_words = set(word_tokenize(sentence.lower()))
    word_freq_per_sentence[idx] = (sentence, len(unique_words))

sorted_sentences = sorted(word_freq_per_sentence.items(), key=lambda x: x[1][1], reverse=True)

total_unique_words = sum(len(set(word_tokenize(sentence.lower()))) for sentence in sentences)
print("Total number of unique words in the text:", total_unique_words)
print()

for idx, (sentence_num, unique_word_count) in enumerate(sorted_sentences, 1):
    print("Sentence", sentence_num, " ")
    print("Number of unique words:", unique_word_count[1])
    print("Sentence:", unique_word_count[0])
    print()

percentage = float(input("Enter the percentage of sentences to include in the summary (from 1 to 100): ")) / 100
print()

sorted_indices = [idx for idx, _ in sorted_sentences]
n = int(len(sentences) * percentage)
summary_sentences = sorted_indices[:n]
summary_sentences.sort()

summary = ' '.join([sentences[idx - 1] for idx in summary_sentences])
print(summary)
