import numpy as np
import collections
import string
import nltk
import gensim
import data_loader

TAGS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNCTUATIONS = set(list(string.punctuation) + ["``", "''"])


def process_row(row):
    tokens = []
    text: str = row["comment_text"].strip()
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if word in PUNCTUATIONS or word in STOPWORDS:
                continue
            tokens.append(word.lower())

    label_one_hot = np.zeros(len(TAGS))
    for idx, tag in enumerate(TAGS):
        if row[tag] != 0:
            label_one_hot[idx] = 1
    return gensim.models.doc2vec.TaggedDocument(words=tokens, tags=label_one_hot)


def load_train_data_with_dictionary(file_path):
    dictionary = {"PADDING_TOKEN": 0, "UNKNOWN_TOKEN": 1}

    train_df, valid_df = data_loader.load_train_data(file_path)
    train_doc, valid_doc = [], []
    for idx, row in train_df.iterrows():
        doc = process_row(row)

        train_doc.append(doc)
        for token in doc.words:
            if token not in dictionary:
                dictionary[token] = len(dictionary)

    for idx, row in valid_df.iterrows():
        doc = process_row(row)
        valid_doc.append(doc)

    return train_doc, valid_doc, dictionary


def load_raw_train_data(file_path):
    train_doc, valid_doc, dictionary = load_train_data_with_dictionary(file_path)

    x_tr, y_tr = [], []
    for doc in train_doc:
        x_tr.append(doc.words)
        y_tr.append(doc.tags)

    x_va, y_va = [], []
    for doc in valid_doc:
        x_va.append(doc.words)
        y_va.append(doc.tags)

    return x_tr, y_tr, x_va, y_va


def load_processed_train_data(file_path):
    train_doc, valid_doc, dictionary = load_train_data_with_dictionary(file_path)

    x_tr, y_tr = [], []
    for doc in train_doc:
        x_tr.append(list(map(lambda x: dictionary.get(x, 1), doc.words)))
        y_tr.append(doc.tags)

    x_va, y_va = [], []
    for doc in valid_doc:
        x_va.append(list(map(lambda x: dictionary.get(x, 1), doc.words)))
        y_va.append(doc.tags)

    return x_tr, y_tr, x_va, y_va


def seq2bow(seq):
    counter = collections.defaultdict(int)
    for token in seq:
        counter[token] += 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    X_tr, Y_tr, X_va, Y_va = load_processed_train_data("./resources/train.csv")
    print(X_tr[:10])
