import data_process
import util
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import logging

logging.basicConfig(level=logging.INFO)


def nb_onehot():
    X_tr, Y_tr, X_va, Y_va, dictionary, X_te, id_list = util.create_or_load_data(freq_threshold=50)

    Y_te_pred_list = []
    sum_auc_va = 0.0
    for i in range(Y_tr.shape[1]):
        nb = BernoulliNB()

        j = 0
        batch_size = 10000
        while j < len(X_tr):
            end = min(j + batch_size, len(X_tr) - 1)
            batch = [data_process.seq2onehot(seq, dictionary) for seq in X_tr[j:end]]
            nb.partial_fit(batch, Y_tr[j:end, i], classes=[0, 1])
            j += batch_size

        logging.info("Finish training")

        Y_va_pred = []
        j = 0
        while j < len(X_va):
            end = min(j + batch_size, len(X_va))
            batch = [data_process.seq2onehot(seq, dictionary) for seq in X_va[j:end]]
            Y_va_pred.extend(nb.predict_proba(batch))
            j += batch_size

        auc_va = util.auc(Y_va[:, i], Y_va_pred)
        logging.info("tag{}, valid auc: ".format(i) + str(auc_va))
        sum_auc_va += auc_va

        Y_te_pred = []
        j = 0
        while j < len(X_te):
            end = min(j + batch_size, len(X_te))
            batch = [data_process.seq2onehot(seq, dictionary) for seq in X_te[j:end]]
            Y_te_pred.extend(nb.predict_proba(batch))
            j += batch_size
        Y_te_pred_list.append(Y_te_pred)

    logging.info("Avg auc: {}".format(sum_auc_va / Y_tr.shape[1]))

    util.submission(Y_te_pred_list, id_list)


def nb_count():
    X_tr, Y_tr, X_va, Y_va, dictionary, X_te, id_list = util.create_or_load_data(freq_threshold=50)

    Y_te_pred_list = []
    sum_auc_va = 0.0
    for i in range(Y_tr.shape[1]):
        nb = MultinomialNB()

        j = 0
        batch_size = 10000
        while j < len(X_tr):
            end = min(j + batch_size, len(X_tr) - 1)
            batch = [data_process.seq2counts(seq, dictionary) for seq in X_tr[j:end]]
            nb.partial_fit(batch, Y_tr[j:end, i], classes=[0, 1])
            j += batch_size

        logging.info("Finish training")

        Y_va_pred = []
        j = 0
        while j < len(X_va):
            end = min(j + batch_size, len(X_va))
            batch = [data_process.seq2counts(seq, dictionary) for seq in X_va[j:end]]
            Y_va_pred.extend(nb.predict_proba(batch))
            j += batch_size

        auc_va = util.auc(Y_va[:, i], Y_va_pred)
        logging.info("tag{}, valid auc: ".format(i) + str(auc_va))
        sum_auc_va += auc_va

        Y_te_pred = []
        j = 0
        while j < len(X_te):
            end = min(j + batch_size, len(X_te))
            batch = [data_process.seq2counts(seq, dictionary) for seq in X_te[j:end]]
            Y_te_pred.extend(nb.predict_proba(batch))
            j += batch_size
        Y_te_pred_list.append(Y_te_pred)

    logging.info("Avg auc: {}".format(sum_auc_va / Y_tr.shape[1]))

    util.submission(Y_te_pred_list, id_list)


if __name__ == '__main__':
    nb_count()
