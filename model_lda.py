import data_process
import logging
import numpy as np
import pickle
import csv
from gensim.models.ldamodel import LdaModel
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_auc_score

MODEL_PATH = "./save/lda{}.model"
TRAIN_DATA_PATH = "./save/processed_train.pkl"
TEST_DATA_PATH = "./save/processed_test.pkl"

logging.basicConfig(level=logging.INFO)


def lda_input_transform(seqs: list):
    trans = [data_process.seq2bow(seq) for seq in seqs]
    return trans


def print_topic_terms(model: LdaModel):
    for topic_id in range(model.num_topics):
        top_list = model.get_topic_terms(topic_id)
        for idx, value in top_list:
            print(topic_id, idx2token[idx], value)


def lda_topic_vectors(model: LdaModel, corpus):
    output = []
    for i, corpora in enumerate(corpus):
        vector = np.zeros(model.num_topics, dtype="float32")
        for topic_id, prob in model[corpora]:
            vector[topic_id] = prob
        vector /= np.linalg.norm(vector)
        output.append(vector)
    return output


def auc(Y, Y_pred):
    auc_sum = 0
    for label, y_pred in zip(Y, Y_pred):
        y_hat = np.zeros(len(y_pred))
        y_hat[label] = 1
        auc_sum += roc_auc_score(y_hat, y_pred)
    return auc_sum / len(Y)


def submission(pred_list, id_list, output_path="submission.csv"):
    with open(output_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(["id"] + data_process.TAGS)

        for idx in range(len(id_list)):
            row = [id_list[idx]]
            for j in range(len(data_process.TAGS)):
                positive_prob = pred_list[j][idx][1]
                row.append(positive_prob)
            csv_writer.writerow(row)


if __name__ == '__main__':
    try:
        with open(TRAIN_DATA_PATH, 'rb') as file:
            X_tr, Y_tr, X_va, Y_va, dictionary = pickle.load(file)
    except Exception as e:
        logging.warning(e)
        X_tr, Y_tr, X_va, Y_va, dictionary = data_process.load_processed_train_data("./resources/train.csv",
                                                                                    freq_threshold=10)
        with open(TRAIN_DATA_PATH, 'wb') as file:
            pickle.dump((X_tr, Y_tr, X_va, Y_va, dictionary), file, pickle.HIGHEST_PROTOCOL)
        logging.info("Data processed")

    logging.info("Data loaded")
    logging.info("Dictionary size: {}".format(len(dictionary)))

    try:
        with open(TEST_DATA_PATH, 'rb') as file:
            X_te, id_list = pickle.load(file)
    except Exception as e:
        logging.warning(e)
        X_te, id_list = data_process.load_processed_test_data_feature_only("./resources/test.csv", dictionary)
        with open(TEST_DATA_PATH, 'wb') as file:
            pickle.dump((X_te, id_list), file, pickle.HIGHEST_PROTOCOL)
        logging.info("Test data processed")

    logging.info("Test data loaded")

    idx2token = {v: k for k, v in dictionary.items()}

    corpus_tr = lda_input_transform(X_tr)
    corpus_va = lda_input_transform(X_va)
    corpus_te = lda_input_transform(X_te)

    NUM_TOPICS = 100
    try:
        lda = LdaModel.load(MODEL_PATH.format(NUM_TOPICS))
    except:
        lda = LdaModel(corpus_tr, num_topics=NUM_TOPICS, dtype=np.float64, minimum_probability=1e-6,
                       minimum_phi_value=1e-6, passes=4)
        lda.save(MODEL_PATH.format(NUM_TOPICS))
        logging.info("Training finish")

    logging.info("Model loaded")

    X_tr = lda_topic_vectors(lda, corpus_tr)
    Y_tr = np.asarray(Y_tr, dtype="int32")

    X_va = lda_topic_vectors(lda, corpus_va)
    Y_va = np.asarray(Y_va, dtype="int32")

    X_te = lda_topic_vectors(lda, corpus_te)

    Y_te_pred_list = []
    for i in range(Y_tr.shape[1]):
        lr = LogisticRegression()
        lr.fit(X_tr, Y_tr[:, i])
        Y_va_pred = lr.predict_proba(X_va)
        print("tag{}, valid auc:".format(i), auc(Y_va[:, i], Y_va_pred))

        Y_te_pred = lr.predict_proba(X_te)
        Y_te_pred_list.append(Y_te_pred)

    submission(Y_te_pred_list, id_list)
