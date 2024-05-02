
Summary:
During this lab, we explored various techniques in Natural Language Processing (NLP) including rule-based methods, regular expressions, and word embeddings. The main objectives were to generate a bill from a text using regex and to apply different word embedding techniques on a given dataset.
In the first part of the lab, we successfully implemented a Python code using regex to generate a bill from a given text. We extracted product names, quantities, and prices using regular expressions and formatted them into a bill format.
For the second part, we explored different word embedding techniques:
One-Hot Encoding, Bag of Words, and TF-IDF:
We applied these techniques to convert text data into numerical vectors.
One-Hot Encoding represents each word as a binary vector, Bag of Words counts the frequency of each word, and TF-IDF gives importance to words based on their frequency in the corpus.
Word2Vec (Skip-gram and CBOW):
We used the Word2Vec model to learn word embeddings from our dataset.
Skip-gram predicts context words given a target word, while CBOW predicts the target word given context words.
GloVe and FastText:
We utilized pre-trained GloVe and FastText models to obtain word embeddings.
GloVe (Global Vectors for Word Representation) and FastText both generate word vectors based on co-occurrence statistics and subword information.
Finally, we visualized the encoded and vectorized vectors using t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm to understand their distribution in a lower-dimensional space.
Report:
Lab Report: NLP Rule-based, Regex, and Word Embedding Techniques
Objective:
The main objective of this lab was to gain practical experience in NLP techniques, including rule-based methods, regular expressions, and word embeddings. Specifically, we aimed to generate a bill from a given text using regex and apply various word embedding techniques on a provided dataset.
Part 1: Rule-Based NLP and Regex:
We successfully implemented a Python code using regular expressions to extract product information from a text and generate a bill. The code parsed the text, identified product names, quantities, and prices, and formatted them into a bill format.
Part 2: Word Embedding:
One-Hot Encoding, Bag of Words, TF-IDF:
We applied these techniques to convert text data into numerical vectors.
One-Hot Encoding represented each word as a binary vector, Bag of Words counted the frequency of each word, and TF-IDF gave importance to words based on their frequency in the corpus.
Word2Vec:
We used Word2Vec with Skip-gram and CBOW architectures to learn word embeddings.
Skip-gram predicted context words given a target word, while CBOW predicted the target word given context words.
GloVe and FastText:
We utilized pre-trained GloVe and FastText models to obtain word embeddings.
GloVe and FastText both provided word vectors based on co-occurrence statistics and subword information.
Visualization:
We visualized the encoded and vectorized vectors using t-SNE algorithm to observe their distribution in a lower-dimensional space. This helped us understand the relationships between words in the embedding space.
Conclusion:
Through this lab, we gained hands-on experience in NLP techniques, regex, and word embeddings. We learned how to extract meaningful information from text using rule-based methods and regex, and how to represent words numerically using various embedding techniques. Visualizing word embeddings provided insights into the semantic relationships between words. Overall, this lab enhanced our understanding of NLP fundamentals and their practical applications.
Tools Used:
Google Colab
GitLab/GitHub
SpaCy
NLTK
Scikit-learn (Sklearn)

