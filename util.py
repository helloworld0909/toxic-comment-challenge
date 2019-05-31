import csv
import numpy as np
from sklearn.metrics import roc_auc_score
import data_process
import pickle
import logging


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


def create_or_load_data(freq_threshold=0):
    train_path = data_process.TRAIN_DATA_PATH.format(freq_threshold)
    test_path = data_process.TEST_DATA_PATH.format(freq_threshold)
    try:
        with open(train_path, 'rb') as file:
            X_tr, Y_tr, X_va, Y_va, dictionary = pickle.load(file)
    except Exception as e:
        logging.warning(e)
        X_tr, Y_tr, X_va, Y_va, dictionary = data_process.load_processed_train_data("./resources/train.csv",
                                                                                    freq_threshold=freq_threshold)
        with open(train_path, 'wb') as file:
            pickle.dump((X_tr, Y_tr, X_va, Y_va, dictionary), file, pickle.HIGHEST_PROTOCOL)
        logging.info("Data processed")

    logging.info("Data loaded")
    logging.info("Dictionary size: {}".format(len(dictionary)))

    try:
        with open(test_path, 'rb') as file:
            X_te, id_list = pickle.load(file)
    except Exception as e:
        logging.warning(e)
        X_te, id_list = data_process.load_processed_test_data_feature_only("./resources/test.csv", dictionary)
        with open(test_path, 'wb') as file:
            pickle.dump((X_te, id_list), file, pickle.HIGHEST_PROTOCOL)
        logging.info("Test data processed")

    logging.info("Test data loaded")

    Y_tr = np.asarray(Y_tr, dtype="int32")
    Y_va = np.asarray(Y_va, dtype="int32")

    return X_tr, Y_tr, X_va, Y_va, dictionary, X_te, id_list


def load_word_embedding(filepath, dim=100):
    word2vector = {'PADDING_TOKEN': np.zeros(dim), 'UNKNOWN_TOKEN': np.random.uniform(-0.25, 0.25, dim)}
    with open(filepath, 'r', encoding='utf-8') as embeddingFile:
        for line in embeddingFile:
            data_tuple = line.rstrip('\n').split(' ')
            token = data_tuple[0]
            vector = data_tuple[1:]
            word2vector[token] = vector
    return word2vector
