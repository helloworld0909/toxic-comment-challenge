import data_process
import util
import logging
import numpy as np
from gensim.models.ldamodel import LdaModel
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_PATH = "./save/lda{}.model"

logging.basicConfig(level=logging.INFO)


def lda_input_transform(seqs: list):
    trans = [data_process.seq2bow(seq) for seq in seqs]
    return trans


def print_topic_terms(model: LdaModel):
    for topic_id in range(model.num_topics):
        top_list = model.get_topic_terms(topic_id)
        print(topic_id, [idx2token[idx] for idx, value in top_list])


def lda_topic_vectors(model: LdaModel, corpus):
    output = []
    for i, corpora in enumerate(corpus):
        vector = np.zeros(model.num_topics, dtype="float32")
        for topic_id, prob in model[corpora]:
            vector[topic_id] = prob
        vector /= np.linalg.norm(vector)
        output.append(vector)
    return output


if __name__ == '__main__':
    X_tr, Y_tr, X_va, Y_va, dictionary, X_te, id_list = util.create_or_load_data(freq_threshold=10)

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

    print_topic_terms(model=lda)

    X_tr = lda_topic_vectors(lda, corpus_tr)
    X_va = lda_topic_vectors(lda, corpus_va)
    X_te = lda_topic_vectors(lda, corpus_te)

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    logging.info("LDA vectors loaded")

    Y_te_pred_list = []
    for i in range(Y_tr.shape[1]):
        classifier = XGBClassifier(max_depth=5, verbosity=2)

        classifier.fit(X_tr, Y_tr[:, i])
        Y_va_pred = classifier.predict_proba(X_va)
        print("tag{}, valid auc:".format(i), util.auc(Y_va[:, i], Y_va_pred))

        Y_te_pred = classifier.predict_proba(X_te)
        Y_te_pred_list.append(Y_te_pred)

    util.submission(Y_te_pred_list, id_list, output_path="submission_xgb.csv")
