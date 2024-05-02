Part 1: Rule Based NLP and Regex


In this part, we are creating a Python code that generates a bill from a given text using regular expressions (regex).

Regular Expressions (Regex):
Regular expressions are sequences of characters that define a search pattern.
We define patterns to match specific parts of the input text, such as product names and prices.
Generating Bill Function:
We define a function generate_bill(text) that takes the input text as a parameter.
Inside the function, we define regex patterns to extract product names and prices from the text.
We use re.search() to find matches for each pattern in the text.
If a match is found, we extract the relevant information and store it in lists.
We then create a Pandas DataFrame to organize the extracted data into a bill format.
Finally, we calculate the total price for each product by multiplying the quantity with the unit price.
Part 2: Word Embedding
In this part, we apply various word embedding techniques on a sample dataset and visualize the vectors using t-SNE.

One-Hot Encoding:
We use CountVectorizer with binary=True to create one-hot encoded vectors.
Each word is represented as a binary vector, where each dimension corresponds to a unique word in the vocabulary.
Bag of Words (BoW):
We use CountVectorizer to create bag of words vectors.
Each word is represented as a count of its occurrences in the document.
TF-IDF (Term Frequency-Inverse Document Frequency):
We use TfidfVectorizer to create TF-IDF vectors.
TF-IDF measures the importance of a word in a document relative to the entire corpus.
Word2Vec:
We use Word2Vec model from Gensim to generate word embeddings.
We train the model on our sample dataset using Skip-Gram approach.
Skip-Gram predicts the context words given a target word.
GloVe:
We load a pre-trained GloVe model using Gensim.
GloVe stands for Global Vectors for Word Representation.
It learns word vectors by factorizing the co-occurrence matrix of words.
FastText:
We use FastText model from Gensim to generate word embeddings.
FastText breaks words into n-grams and represents each word as a sum of its n-gram vectors.
t-SNE Visualization:
We use t-distributed Stochastic Neighbor Embedding (t-SNE) to reduce the dimensionality of the vectors.
t-SNE is a technique for dimensionality reduction that is particularly well-suited for visualizing high-dimensional data.
We plot the vectors in a 2D space to visualize their relationships.
Conclusion:
Each word embedding technique captures different aspects of the data.
One-hot encoding and Bag of Words are simple and interpretable but lose semantic information.
TF-IDF considers the importance of words but still lacks context.
Word2Vec, GloVe, and FastText capture semantic relationships between words and are more suitable for NLP tasks.
Visualization with t-SNE helps us understand the relationships between words in a lower-dimensional space.
By combining these techniques, we can represent text data in a format suitable for various NLP tasks like classification, clustering, and information retrieval
