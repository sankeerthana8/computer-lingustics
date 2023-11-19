from typing import List
from nltk import word_tokenize # the nltk word tokenizer
from spacy.lang.en import English  # for the spacy tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json, math
from collections import Counter

# spacy things


# gpt2 tokenizer

# gpt2 model

# softmax


def load_corpus(filename: str):
    corpus = None
    return corpus


def nltk_tokenize(sentence: str):
    nltk_tokenized = None
    return nltk_tokenized # must be type list of strings


def spacy_tokenize(sentence: str):
    spacy_tokenized = None
    return spacy_tokenized # must be type list of strings

def tokenize(sentence: str):
    # wrapper around whichever tokenizer you liked better
    wrapped_output = None
    return wrapped_output


# part of solution to 2a
def count_bigrams(corpus: list):
    bigrams_frequencies = None
    return bigrams_frequencies


# part of solution to 2a
def count_trigrams(corpus: list):
    trigrams_frequencies = None
    return trigrams_frequencies


# part of solution to 2b
def bigram_frequency(bigram: str, bigram_frequency_dict: dict):
    frequency_of_bigram = None
    return frequency_of_bigram


# part of solution to 2c
def trigram_frequency(trigram: str, trigram_frequency_dict: dict):
    frequency_of_trigram = None
    return frequency_of_trigram
    

# part of solution to 2d
def get_total_frequency(ngram_frequencies: dict):
    # compute the frequency of all ngrams from dictionary of counts
    total_frequency = None
    return total_frequency


# part of solution to 2e
def get_probability(
        ngram: str,
        ngram_frequencies: dict):
    probability = None
    return probability


# part of solution to 3a
def forward_transition_probability(
        seq_of_three_tokens: list,
        trigram_counts: dict,
        bigram_counts: dict
        ):
    fw_prob = None 
    return fw_prob


# part of solution to 3b
def backward_transition_probability(
        seq_of_three_tokens: list,
        trigram_counts: dict,
        bigram_counts: dict
        ):
    bw_prob = None
    return bw_prob


# part of solution to 3c
def compare_fw_bw_probability(fw_prob: float, bw_prob: float):
    equivalence_test = None
    return equivalence_test


# part of solution to 3d
def sentence_likelihood(
    sentence,  # an arbitrary string
    bigram_counts,   # the output of count_bigrams
    trigram_counts   # the output of count_trigrams
    ):
    likelihood = None
    return likelihood


# 4a
def neural_tokenize(sentence: str):
    tokenizer_output = gpt2_tokenizer(
        sentence, return_tensors="pt"
        ) #Encode the text into gpt2 tokens
    return tokenizer_output


# 4b
def neural_logits(tokenizer_output):
    logits = None
    return logits 


# 4c
def normalize_probability(logits):
    softmax_logits = None
    return softmax_logits


# 4d.i
def neural_fw_probability(
    softmax_logits,
    tokenizer_output
    ):
    probabilities = None
    return probabilities


# 4d.ii
def neural_likelihood(diagonal_of_probs):
    likelihood = None
    return likelihood
