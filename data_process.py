import numpy as np
import collections
import string
import nltk
import gensim
import data_loader

TAGS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNCTUATIONS = set(list(string.punctuation) + ["``", "''"])

TRAIN_DATA_PATH = "./save/processed_train_{}.pkl"
TEST_DATA_PATH = "./save/processed_test_{}.pkl"


def process_row(row):
    text: str = row["comment_text"]
    tokens = tokenize(text)
    label_one_hot = np.zeros(len(TAGS))
    for idx, tag in enumerate(TAGS):
        if row[tag] != 0:
            label_one_hot[idx] = 1
    return gensim.models.doc2vec.TaggedDocument(words=tokens, tags=label_one_hot)


def tokenize(text):
    tokens = []
    for sent in nltk.sent_tokenize(text.strip()):
        for word in nltk.word_tokenize(sent):
            tokens.append(word.lower())
    return tokens


def load_train_data_with_dictionary(file_path, freq_threshold=0):
    dictionary = {"PADDING_TOKEN": 0, "UNKNOWN_TOKEN": 1}
    word_freq = collections.defaultdict(int)

    train_df, valid_df = data_loader.load_train_data(file_path)
    train_doc, valid_doc = [], []
    for idx, row in train_df.iterrows():
        doc = process_row(row)

        train_doc.append(doc)
        for token in doc.words:
            word_freq[token] += 1

    for word, freq in word_freq.items():
        if freq >= freq_threshold:
            dictionary[word] = len(dictionary)

    for idx, row in valid_df.iterrows():
        doc = process_row(row)
        valid_doc.append(doc)

    return train_doc, valid_doc, dictionary


def load_raw_train_data(file_path, freq_threshold=0):
    train_doc, valid_doc, dictionary = load_train_data_with_dictionary(file_path, freq_threshold)

    x_tr, y_tr = [], []
    for doc in train_doc:
        x_tr.append(doc.words)
        y_tr.append(doc.tags)

    x_va, y_va = [], []
    for doc in valid_doc:
        x_va.append(doc.words)
        y_va.append(doc.tags)

    return x_tr, y_tr, x_va, y_va, dictionary


def load_processed_train_data(file_path, freq_threshold=0):
    train_doc, valid_doc, dictionary = load_train_data_with_dictionary(file_path, freq_threshold)

    x_tr, y_tr = [], []
    for doc in train_doc:
        x_tr.append(list(map(lambda x: dictionary.get(x, 1), doc.words)))
        y_tr.append(doc.tags)

    x_va, y_va = [], []
    for doc in valid_doc:
        x_va.append(list(map(lambda x: dictionary.get(x, 1), doc.words)))
        y_va.append(doc.tags)

    return x_tr, y_tr, x_va, y_va, dictionary


def load_processed_test_data_feature_only(file_path, dictionary):
    test_df = data_loader.load_test_data_feature_only(file_path)
    x_te, id_list = [], []

    for idx, row in test_df.iterrows():
        text = row["comment_text"]
        tokens = tokenize(text)
        x_te.append(list(map(lambda x: dictionary.get(x, 1), tokens)))
        id_list.append(row["id"])

    return x_te, id_list


def seq2bow(seq):
    counter = collections.defaultdict(int)
    for token in seq:
        counter[token] += 1
    return list(counter.items())


def seq2onehot(seq, dictionary):
    vector = np.zeros(len(dictionary))
    for token in seq:
        vector[token] = 1
    return vector


if __name__ == '__main__':
    X_tr, Y_tr, X_va, Y_va, dic = load_processed_train_data("./resources/train.csv")
    print(X_tr[:10])
