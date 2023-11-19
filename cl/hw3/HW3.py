import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_scores(data):
    unigram_precision_scores = []
    unigram_recall_scores = []
    for i, row in data.iterrows():
        gold_sentence = row['eng_gold']
        translated_sentence = row['fin_translated']
        
        # Tokenize the sentences into lists of words
        gold_tokens = nltk.word_tokenize(gold_sentence.lower())
        translated_tokens = nltk.word_tokenize(translated_sentence.lower())
        
        # Compute unigram precision and recall scores
        unigram_precision = len(set(translated_tokens).intersection(set(gold_tokens))) / len(translated_tokens)
        unigram_recall = len(set(translated_tokens).intersection(set(gold_tokens))) / len(gold_tokens)
        
        unigram_precision_scores.append(unigram_precision)
        unigram_recall_scores.append(unigram_recall)
    
    return unigram_precision_scores, unigram_recall_scores

# Load the data
import pandas as pd
data = pd.read_csv('eng-via-fin_translations.tsv', sep='\t')

# Compute the scores
unigram_precision_scores, unigram_recall_scores = compute_scores(data)

# Compute the mean scores
mean_unigram_precision = sum(unigram_precision_scores) / len(unigram_precision_scores)
mean_unigram_recall = sum(unigram_recall_scores) / len(unigram_recall_scores)

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# Compute the embeddings for the gold and translated sentences
gold_embeddings = model.encode(data['eng_gold'].tolist())
translated_embeddings = model.encode(data['fin_translated'].tolist())

cosine_similarities = cosine_similarity(gold_embeddings, translated_embeddings)

# Compute the average cosine similarity
mean_cosine_similarity = np.mean(cosine_similarities)

# Compute the number of translations with a cosine similarity of 1
num_cosine_similarities_of_1 = np.sum(cosine_similarities == 1)

#Q2
# Pair 1
gold_pronouns_pair1 = ['they', 'their', 'them']
translated_pronouns_pair1 = ['he', 'his', 'him']

# Pair 2
gold_pronouns_pair2 = ['she', 'her']
translated_pronouns_pair2 = ['he', 'him']

# Pair 3
gold_pronouns_pair3 = ['they', 'their', 'them']
translated_pronouns_pair3 = ['they', 'their', 'them']

num_sentences = len(data)
same_num_pronouns = 0
#b
for i in range(num_sentences):
    gold_sentence = data.loc[i, 'eng_gold']
    translated_sentence = data.loc[i, 'fin_translated']
    
    num_gold_pronouns = sum([1 for word in gold_sentence.split() if word.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']])
    num_translated_pronouns = sum([1 for word in translated_sentence.split() if word.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']])
    
    if num_gold_pronouns == num_translated_pronouns:
        same_num_pronouns += 1

proportion_same_num_pronouns = round(same_num_pronouns / num_sentences, 3)
#c
proportions_match = []

for i in range(num_sentences):
    gold_sentence = data.loc[i, 'eng_gold']
    translated_sentence = data.loc[i, 'fin_translated']
    
    num_gold_pronouns = sum([1 for word in gold_sentence.split() if word.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']])
    num_translated_pronouns = sum([1 for word in translated_sentence.split() if word.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']])
    
    if num_gold_pronouns == num_translated_pronouns:
        gold_pronouns = [word.lower() for word in gold_sentence.split() if word.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']]
        translated_pronouns = [word.lower() for word in translated_sentence.split() if word.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']]
        
        num_pronouns = num_gold_pronouns
        num_matches = 0
        
        for j in range(num_pronouns):
            if gold_pronouns[j] == translated_pronouns[j]:
                num_matches += 1
                
        proportion_matches = num_matches / num_pronouns
        proportions_match.append(proportion_matches)
        
mean_proportion_matches = round(np.mean(proportions_match), 3)
print(mean_proportion_matches)
print(f"The mean proportion of matching pronouns in the gold and translated sentences is {mean_proportion_matches}. This indicates the overall accuracy of pronoun translation in the dataset.")

#c2
masculine_proportions = []
feminine_proportions = []
they_proportions = []

for i in range(num_sentences):
    gold_sentence = data.loc[i, 'eng_gold']
    translated_sentence = data.loc[i, 'fin_translated']
    
    num_gold_pronouns = sum([1 for word in gold_sentence.split() if word.lower() in ['he', 'him', 'his']])
    num_translated_pronouns = sum([1 for word in translated_sentence.split() if word.lower() in ['he', 'him', 'his']])
    
    if num_gold_pronouns == num_translated_pronouns:
        gold_pronouns = [word.lower() for word in gold_sentence.split() if word.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']]
        translated_pronouns = [word.lower() for word in translated_sentence.split() if word.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']]
        
        num_pronouns = num_gold_pronouns
        num_matches = 0
        
        for j in range(num_pronouns):
            if gold_pronouns[j] == translated_pronouns[j]:
                num_matches += 1
                
        proportion_matches = num_matches / num_pronouns
        
        if 'he' in gold_pronouns:
            masculine_proportions.append(proportion_matches)
        elif 'she' in gold_pronouns:
            feminine_proportions.append(proportion_matches)
        else:
            they_proportions.append(proportion_matches)

mean_masculine_proportion = round(np.mean(masculine_proportions), 3)
mean_feminine_proportion = round(np.mean(feminine_proportions), 3)
mean_they_proportion = round(np.mean(they_proportions), 3)

print(f"Mean proportion of correct translations for masculine pronouns: {mean_masculine_proportion}")
print(f"Mean proportion of correct translations for feminine pronouns: {mean_feminine_proportion}")


