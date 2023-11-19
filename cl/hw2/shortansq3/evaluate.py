from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import numpy as np
from collections import Counter
import HW2
from time import time
import pandas as pd

TRAINING_CORPUS = 'training_sentences.txt'
# TEST_CORPUS = 'test_sentences.txt'

def main():
    print("Loading corpus")
    train_sentences, DOs, IOs, train_ys = load_corpus(TRAINING_CORPUS)
    # get matcher labels
    print("Checking matcher accuracy")
    matcher_classifications = get_classifications(train_sentences)
    matcher_accuracy = [matcher_classifications[i]==train_ys[i] for i, _ in enumerate(train_ys)]
    # get features
    print("Extracting features")
    speed_features_start = time()
    DO_nums, IO_nums, DO_string_features_, IO_string_features_ = extract_features_all_sentences(train_sentences, DOs, IOs)
    speed_features = time() - speed_features_start
    DO_cv, IO_cv, DO_string_features_, IO_string_features_ = vectorize_string_features(DO_string_features_, IO_string_features_, mode='train')
    train_Xs = concatenate_features(DO_nums, DO_string_features_, IO_nums, IO_string_features_)
    # train model
    print("Training model")
    train_Xs, train_ys = remove_nans(train_Xs, train_ys)
    trained_model = train_model(train_Xs, train_ys)
    train_predictions = trained_model.predict(train_Xs)
    training_f1 = f1_score(train_predictions, train_ys, pos_label='PO')
    print(f"Your F1 score for predicting sentence order on the training data is {training_f1}")
    # print("Testing model")
    # test_sentences, DOs_test, IOs_test, test_ys = load_corpus(TEST_CORPUS)
    # DO_nums_test, IO_nums_test, DO_string_features_test, IO_string_features_test = extract_features_all_sentences(test_sentences, DOs_test, IOs_test)
    # DO_string_features_test_, IO_string_features_test_ = vectorize_string_features(DO_string_features_test, IO_string_features_test, mode='test', DO_cv=DO_cv, IO_cv=IO_cv)
    # test_Xs = concatenate_features(DO_nums_test, DO_string_features_test_, IO_nums_test, IO_string_features_test_)
    # test_Xs, test_ys = remove_nans(test_Xs, test_ys)
    # test_predictions = trained_model.predict(test_Xs)
    # print(test_predictions)
    # test_f1 = f1_score(test_predictions, test_ys, pos_label='PO')
    # print(f"Your F1 score for predicting sentence order on the test data is {test_f1}")


def load_corpus(filename):
    data = pd.read_csv(filename, sep="\t")
    sentences = data['sentence'].tolist()
    labels = data['structure'].tolist()
    direct_objects = data['direct_object']
    indirect_objects = data['indirect_object']
    return sentences, direct_objects, indirect_objects, labels


def extract_numeric_features(NP, sentence):
    int_feature = HW2.extract_feature_1(NP, sentence)
    float_feature = HW2.extract_feature_3(NP, sentence)
    return int_feature, float_feature


def extract_string_features(NP, sentence):
    string_features = HW2.extract_feature_2(NP, sentence)
    return string_features


def concatenate_features(
        DO_numeric_features,
        DO_string_features,
        PO_numeric_features,
        PO_string_features
        ):
    DO_features = np.concatenate([DO_string_features, DO_numeric_features], axis=1)
    PO_features = np.concatenate([PO_string_features, PO_numeric_features], axis=1)
    all_features = np.concatenate([DO_features, PO_features], axis=1)
    return all_features


def extract_all_features_sentence(s, DO_NP, IO_NP):
    if DO_NP is None:
        DO_NP = ''
    if IO_NP is None:
        IO_NP = ''
    DO_numeric_feature = extract_numeric_features(DO_NP, s)
    IO_numeric_feature = extract_numeric_features(IO_NP, s)
    DO_string_feature = extract_string_features(DO_NP, s)
    IO_string_feature = extract_string_features(IO_NP, s)
    return DO_numeric_feature, DO_string_feature, IO_numeric_feature, IO_string_feature


def extract_features_all_sentences(
        list_of_sentences, DOs, IOs
):
    DO_numeric_features, IO_numeric_features = [], []
    DO_string_features, IO_string_features = [], []
    for i, s in enumerate(list_of_sentences):
        DO, IO = DOs[i], IOs[i]
        DO_numeric_feature, DO_string_feature, IO_numeric_feature, IO_string_feature = extract_all_features_sentence(s, DO, IO)
        DO_numeric_features.append(
            DO_numeric_feature
        )
        IO_numeric_features.append(
            IO_numeric_feature
        )
        DO_string_features.append(
            DO_string_feature
            )
        IO_string_features.append(
            IO_string_feature
        )
    return DO_numeric_features, IO_numeric_features, DO_string_features, IO_string_features


def vectorize_string_feature(
    string_features,
    mode='train',
    cv = None,
):
    if mode=='train':
        cv, string_features_ = vectorize_train(string_features)
        return cv, string_features_
    else:
        string_features_ = vectorize_test(string_features, cv)
        return string_features_


def vectorize_string_features(
        DO_string_features, IO_string_features,
        mode='train',
        DO_cv=None,
        IO_cv=None
        ):
    if mode=='train':
        DO_cv, DO_string_features_ = vectorize_string_feature(DO_string_features, mode)
        IO_cv, IO_string_features_ = vectorize_string_feature(IO_string_features, mode)
        return DO_cv, IO_cv, DO_string_features_, IO_string_features_
    else:
        DO_string_features_ = vectorize_string_feature(DO_string_features, mode, DO_cv)
        IO_string_features_ = vectorize_string_feature(IO_string_features, mode, IO_cv)
        return DO_string_features_, IO_string_features_


def dummy_tokenizer(doc):
    return [doc]


def remove_nans(X_train, y_train):
    imputed = []
    for X in X_train:
        imputed.append(np.nan_to_num(X))
    imputed_array = np.array(imputed)
    imputed_array[imputed_array==None] = 0
    no_nans_X = [
        imputed_array[i]
        for i, x in enumerate(y_train)
        if x!=None
    ]
    no_nans_y = [x for x in y_train if x!=None]
    return no_nans_X, no_nans_y


def vectorize_train(list_of_strings: list):
    vectorizer = CountVectorizer(
        min_df=5, max_df=len(list_of_strings),
        tokenizer=dummy_tokenizer)
    counts = Counter(list_of_strings)
    strings_w_unks = []
    for w in counts:
        if counts[w] < 5:
            strings_w_unks += ['UNK'] * counts[w]
        else:
            strings_w_unks += [w] * counts[w]
    vectorized = np.array(vectorizer.fit_transform(
        strings_w_unks
        ).todense())
    return vectorizer, vectorized


def vectorize_test(list_of_strings, vectorizer):
    strings_w_unks = []
    counts = Counter(list_of_strings)
    for w in counts:
        if counts[w] < 5:
            strings_w_unks += ['UNK'] * counts[w]
        else:
            strings_w_unks += [w] * counts[w]
    vectorized = np.array(vectorizer.transform(
        strings_w_unks
        ).todense())
    return vectorized


def get_classification(sentence):
    y = HW2.get_sentence_structure(sentence)
    return y


def get_classifications(list_of_sentences):
    ys = []
    for s in list_of_sentences:
        ys.append(get_classification(s))
    return ys


def train_model(X_train, y_train):
    clf = MLPClassifier(
        alpha=.05, random_state=1, max_iter=300,
        hidden_layer_sizes=(10, 10,),
        batch_size=1
        ).fit(X_train, y_train)
    return clf


if __name__=="__main__":
    main()