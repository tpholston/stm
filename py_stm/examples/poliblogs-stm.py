# %% [markdown]
# ## Example of STM in use
# This is different than poliblogs.ipynb because we are demonstrating use of metadata

# %%
import numpy as np
import pandas as pd
from patsy import dmatrix
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from py_stm.stm import StmModel
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models

# %%
poliblogs = pd.read_csv("test_data/poliblogs2008.csv", )
poliblogs = poliblogs.loc[:, ~poliblogs.columns.str.contains('^Unnamed')]

poliblogs.head()

# %%
print(f"There are {len(poliblogs)} many documents in the poliblogs dataset")

# %%
nltk.download('stopwords')  # Download the stopwords

# %%
def preprocess_text(text, min_len):
    tokens = simple_preprocess(text, deacc=True, min_len=min_len)  # deacc=True removes punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens


# %%
# Split the text and metadata for training
train_text, test_text, train_metadata, test_metadata = train_test_split(
    poliblogs.documents, poliblogs[['rating', 'day', 'blog']], test_size=0.8, random_state=42
)

# Apply the preprocessing function to the text data
processed_train_text = [preprocess_text(text, min_len=3) for text in train_text]
processed_test_text = [preprocess_text(text, min_len=3) for text in test_text]

# Create the training dictionary
dictionary = Dictionary(processed_train_text)

# Filter extremes (remove tokens that appear in less than 10 documents, or more than 50% of the documents)
dictionary.filter_extremes(no_below=10, no_above=0.5)

# Create the training corpus (bag of words representation)
corpus_train = [dictionary.doc2bow(text) for text in processed_train_text]

# %%
from scipy.sparse import csr_matrix
corpus_train[190]
doc_idx, word_idx, count = [], [], []

for i, doc in enumerate(corpus_train):
	for word, freq in doc:
		doc_idx.append(i)
		word_idx.append(word)
		count.append(freq)

a = csr_matrix((count, (doc_idx, word_idx)))

wprob = np.sum(a, axis=0)
wprob = wprob / np.sum(wprob)    
wprob = np.array(wprob)

wprob.flatten()

# %%
# A user could precompute the prevalence themselves like so
from patsy import dmatrix

prevalence = dmatrix("~rating+cr(day, df=3)", data=train_metadata, return_type='dataframe')

# %%
# Train the STM model
num_topics = 5  # Define the number of topics you want to extract
#stm = StmModel(corpus_train, num_topics=num_topics, id2word=dictionary, prevalence=prevalence, passes=2, random_state=420) # intended use 1. prevalence matrix precomputed
#stm = StmModel(corpus_train, num_topics=num_topics, id2word=dictionary, metadata=train_metadata, prevalence="~rating+cr(day, df=3)", passes=2, random_state=420) # intended use 2. metadata dataframe with prevalence formula
stm = StmModel(corpus_train, num_topics=num_topics, id2word=dictionary, metadata=train_metadata, content=train_metadata.loc[:, "blog"], passes=2, random_state=420) # intended use 3. metadata dataframe and content formula

# %%
train_metadata.loc[:, "blog"].head()

# %%



