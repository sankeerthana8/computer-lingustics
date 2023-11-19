# Importing necessary libraries
!pip install transformers
!pip install torch
from typing import List
from nltk import word_tokenize # the nltk word tokenizer
from spacy.lang.en import English  # for the spacy tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json, math
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Defining function to load the corpus
def load_corpus(filename: str) -> List[str]:
    with open(filename,'r') as data:
        corpus = data.read().splitlines()
    return corpus

# Loading the corpus from the given file
corpus = load_corpus('corpus.txt')

# Defining function to tokenize the corpus using nltk
def nltk_tokenize(sentence: str) -> List[str]:
    nltk_tokenized = word_tokenize(sentence)
    return nltk_tokenized

# Tokenizing the corpus using nltk tokenizer and removing the stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [nltk_tokenize(token) for token in corpus if not token.lower() in stop_words]
corpus = [token for sublist in filtered_tokens for token in sublist]

# Defining function to tokenize the corpus using spacy
import spacy
from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm")

def spacy_tokenize(sentence: str) -> List[str]:
    spacy_tokenized = nlp(sentence)
    tokens = [token.text for token in spacy_tokenized]
    return tokens # must be type list of strings

# Tokenizing the corpus using spacy tokenizer
corpus = [spacy_tokenize(token) for token in corpus]
corpus = [token for sublist in corpus for token in sublist]

# Defining function to count bigrams in the corpus
def count_bigrams(corpus: list) -> dict:
    bigram_frequencies = {}
    for i in range(len(corpus)-1):
        bigram = (corpus[i], corpus[i+1])
        if bigram in bigram_frequencies:
            bigram_frequencies[bigram] += 1
        else:
            bigram_frequencies[bigram] = 1
    return bigram_frequencies

# Counting bigrams in the corpus
bigram_frequency_dict = count_bigrams(corpus)

# Defining function to count trigrams in the corpus
def count_trigrams(corpus: list) -> dict:
    trigram_frequencies = {}
    for i in range(len(corpus)-2):
        trigram = (corpus[i], corpus[i+1], corpus[i+2])
        if trigram in trigram_frequencies:
            trigram_frequencies[trigram] += 1
        else:
            trigram_frequencies[trigram] = 1
    return trigram_frequencies

# Counting trigrams in the corpus
trigram_frequency_dict = count_trigrams(corpus)

# Defining function to get frequency of a given bigram
def bigram_frequency(bigram: tuple, bigram_frequency_dict: dict) -> int:
    if bigram in bigram_frequency_dict:
        frequency_of_bigram = bigram_frequency_dict[bigram]
        return frequency_of_bigram
