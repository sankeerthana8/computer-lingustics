import evaluate
import HW2
import pandas as pd
from sklearn.metrics import f1_score
from time import time

TRAINING_CORPUS = 'training_sentences.txt'
TEST_CORPUS = 'test_sentences.txt'

def load_corpus(filename):
    data = pd.read_csv(filename, sep="\t")
    sentences = data['sentence'].tolist()
    DOs=[HW2.extract_direct_object(s) for s in sentences]
    IOs=[HW2.extract_indirect_object(s) for s in sentences]
    labels = data['structure'].tolist()
    return sentences,DOs,IOs, labels

print("Loading corpus")
train_sentences,DOs,IOs ,train_ys = load_corpus(TRAINING_CORPUS)
# get features
print("Extracting features")
speed_features_start = time()
train_Xs, DO_cv, PO_cv, _ = evaluate.extract_features_all_sentences(train_sentences,DOs,IOs)

speed_features = time() - speed_features_start
# get matcher labels
print("Checking matcher accuracy")
matcher_classifications = evaluate.get_classifications(train_sentences)
matcher_accuracy = [matcher_classifications[i]==train_ys[i] for i, _ in enumerate(train_ys)]
# train model
print("Training model")
train_Xs, train_ys = evaluate.remove_nans(train_Xs, train_ys)
trained_model = evaluate.train_model(train_Xs, train_ys)
train_predictions = trained_model.predict(train_Xs)
training_f1 = f1_score(train_predictions, train_ys, pos_label='PO')
print(f"Your F1 score for predicting sentence order on the training data is {training_f1}")
# test accuracy
# print("Testing model")
# test_sentences, test_ys = load_corpus(TEST_CORPUS)
# test_Xs = evaluate.extract_features_all_sentences(test_sentences, mode='test', DO_cv=DO_cv, PO_cv=PO_cv)
# test_Xs, test_ys = evaluate.remove_nans(test_Xs, test_ys)
# test_predictions = trained_model.predict(test_Xs)
# print(test_predictions)
# test_f1 = f1_score(test_predictions, test_ys, pos_label='PO')
# print(f"Your F1 score for predicting sentence order on the test data is {test_f1}")
