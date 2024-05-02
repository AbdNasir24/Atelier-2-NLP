#Part 1: Rule Based NLP and Regex
import re
import pandas as pd

def generate_bill(text):
    # Define patterns for extracting product information
    patterns = [
        (r'Samsung smartphones (\d+) \$ each', 'Samsung smartphones', '150'),
        (r'fresh banana for ([\d,\.]+) dollar a kilogram', 'Banana', '1.2'),
        (r'Hamburger with ([\d,\.]+) dollar', 'Hamburger', '4.5')
    ]

    # Initialize lists to store extracted data
    products = []
    quantities = []
    unit_prices = []

    # Extract product information using regex patterns
    for pattern, product, price in patterns:
        match = re.search(pattern, text)
        if match:
            products.append(product)
            # Remove comma from quantity string and convert to integer
            quantity = int(match.group(1).replace(',', ''))
            quantities.append(quantity)
            unit_prices.append(float(price))

    # Create DataFrame for bill
    bill_df = pd.DataFrame({
        'Product': products,
        'Quantity': quantities,
        'Unit Price': unit_prices
    })

    # Calculate Total Price
    bill_df['Total Price'] = bill_df['Quantity'] * bill_df['Unit Price']

    return bill_df

# Example usage
text = "I bought three Samsung smartphones 150 $ each, four kilos of fresh banana for 1,2 dollar a kilogram and one Hamburger with 4,5 dollar."
bill = generate_bill(text)
print("Generated Bill:\n", bill)

#Part 2: Word Embedding

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.doc2vec import TaggedDocument

# Define sample dataset
data = ["I like to eat bananas.",
        "I enjoy eating apples.",
        "Bananas are my favorite fruit.",
        "I don't like oranges.",
        "Apples and bananas are delicious.",
        "I hate fruits.",
        "Fruits are healthy."]

# 1. One-hot encoding
vectorizer = CountVectorizer(binary=True, lowercase=True)
one_hot_encoded = vectorizer.fit_transform(data).toarray()
print("One-hot Encoded Vectors:\n", one_hot_encoded)

# 2. Bag of words (BoW)
vectorizer = CountVectorizer(lowercase=True)
bow_vectors = vectorizer.fit_transform(data).toarray()
print("Bag of Words Vectors:\n", bow_vectors)

# 3. TF-IDF
vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(data).toarray()
print("TF-IDF Vectors:\n", tfidf_vectors)

# 4. Word2Vec (Skip Gram)
word2vec_sg_model = Word2Vec(sentences=[d.split() for d in data], vector_size=100, sg=1, window=5, min_count=1)
print("Word2Vec (Skip Gram) Vector for 'bananas':\n", word2vec_sg_model.wv['bananas'])

# 5. GloVe
# Load pre-trained GloVe model
glove_model = KeyedVectors.load_word2vec_format("glove.6B.100d.txt", binary=False, no_header=True)

# Test access to vectors
print("GloVe Vector for 'bananas':\n", glove_model['bananas'])

# 6. FastText
fasttext_model = FastText(sentences=[d.split() for d in data], vector_size=100, window=5, min_count=1)
print("FastText Vector for 'bananas':\n", fasttext_model.wv['bananas'])

# Plot vectors using t-SNE
def plot_vectors(vectors, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_vectors = tsne.fit_transform(vectors)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c='blue', edgecolors='k')
    for i, label in enumerate(labels):
        plt.annotate(label, (tsne_vectors[i, 0], tsne_vectors[i, 1]))
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

# Plotting vectors
vectors_to_plot = [one_hot_encoded, bow_vectors, tfidf_vectors,
                   [word2vec_sg_model.wv[word] for word in word2vec_sg_model.wv.key_to_index],
                   [glove_model[word] for word in glove_model.key_to_index],
                   [fasttext_model.wv[word] for word in fasttext_model.wv.key_to_index]]

def plot_vectors(vectors, labels, title):
    vectors = np.array(vectors)  # Convert list of vectors to numpy array
    n_samples = vectors.shape[0]
    perplexity = min(30, n_samples - 1)  # Set perplexity to a maximum of n_samples - 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_vectors = tsne.fit_transform(vectors)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c='blue', edgecolors='k')
    for i, label in enumerate(labels):
        plt.annotate(label, (tsne_vectors[i, 0], tsne_vectors[i, 1]))
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

# Then use it as before
labels = ['One-Hot Encoding', 'Bag of Words', 'TF-IDF', 'Word2Vec (Skip Gram)', 'GloVe', 'FastText']
for i, vectors in enumerate(vectors_to_plot):
    plot_vectors(vectors, labels, f"{labels[i]} Vectors")
