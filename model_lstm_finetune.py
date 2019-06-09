import keras
from keras.preprocessing.sequence import pad_sequences

import logging
import util
from model_lstm import MAX_SENTENCE_LEN

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    X_tr, Y_tr, X_va, Y_va, dictionary, X_te, id_list = util.create_or_load_data(freq_threshold=10)
    idx2token = {v: k for k, v in dictionary.items()}

    X_tr = pad_sequences(X_tr, MAX_SENTENCE_LEN, truncating='post')
    X_va = pad_sequences(X_va, MAX_SENTENCE_LEN, truncating='post')
    X_te = pad_sequences(X_te, MAX_SENTENCE_LEN, truncating='post')

    model_path = "./save/lstm_100.model"

    lstm = keras.models.load_model(model_path)
    lstm.get_layer("word_embedding").trainable = True
    lstm.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    lstm.summary()

    lstm.fit(X_tr, Y_tr, epochs=2, batch_size=64, validation_data=[X_va, Y_va])
    lstm.save(model_path + ".ft")

    Y_te_pred_list = lstm.predict(X_te, verbose=1)
    util.submission2(Y_te_pred_list, id_list, output_path="submission3.csv")
