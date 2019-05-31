import data_process
import util
from sklearn.naive_bayes import BernoulliNB
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    X_tr, Y_tr, X_va, Y_va, dictionary, X_te, id_list = util.create_or_load_data(freq_threshold=200)

    X_tr = [data_process.seq2onehot(seq, dictionary) for seq in X_tr]
    X_va = [data_process.seq2onehot(seq, dictionary) for seq in X_va]
    X_te = [data_process.seq2onehot(seq, dictionary) for seq in X_te]

    Y_te_pred_list = []
    sum_auc_va = 0.0
    for i in range(Y_tr.shape[1]):
        nb = BernoulliNB()
        nb.fit(X_tr, Y_tr[:, i])
        Y_va_pred = nb.predict_proba(X_va)
        auc_va = util.auc(Y_va[:, i], Y_va_pred)
        logging.info("tag{}, valid auc:".format(i), auc_va)
        sum_auc_va += auc_va

        Y_te_pred = nb.predict_proba(X_te)
        Y_te_pred_list.append(Y_te_pred)

    logging.info("Avg auc: {}".format(sum_auc_va / Y_tr.shape[1]))

    util.submission(Y_te_pred_list, id_list)
