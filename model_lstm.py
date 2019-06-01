import keras
from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Input, Masking
from keras.preprocessing.sequence import pad_sequences

import logging
import numpy as np
import util

logging.basicConfig(level=logging.INFO)

MAX_SENTENCE_LEN = 200

PARAMS = {
    'wordEmbeddingDim': 100,
    'lstmOutDim': 100
}


def build_model(word_embedding, params):
    word_input = Input((MAX_SENTENCE_LEN,), name='word_input')
    word_input_masking = Masking(mask_value=0, input_shape=(MAX_SENTENCE_LEN,))(word_input)
    word_embedding = Embedding(
        input_dim=word_embedding.shape[0],
        output_dim=params['wordEmbeddingDim'],
        input_length=MAX_SENTENCE_LEN,
        weights=[word_embedding],
        trainable=False,
        name='word_embedding'
    )(word_input_masking)

    bilstm = Bidirectional(LSTM(params['lstmOutDim'], dropout=0.0, recurrent_dropout=0.0), name='BiLSTM2')(
        word_embedding)

    hidden = Dense(100, activation="relu", name='hidden_layer1')(bilstm)
    hidden = Dense(30, activation="relu", name='hidden_layer2')(hidden)

    output = Dense(6, activation="sigmoid", name='output')(hidden)

    model = Model(inputs=word_input, outputs=output)
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model


if __name__ == '__main__':
    X_tr, Y_tr, X_va, Y_va, dictionary, X_te, id_list = util.create_or_load_data(freq_threshold=10)
    idx2token = {v: k for k, v in dictionary.items()}

    X_tr = pad_sequences(X_tr, MAX_SENTENCE_LEN)
    X_va = pad_sequences(X_va, MAX_SENTENCE_LEN)
    X_te = pad_sequences(X_te, MAX_SENTENCE_LEN)

    model_path = "./save/lstm_{}.model".format(PARAMS['wordEmbeddingDim'])
    try:
        lstm = keras.models.load_model(model_path)
    except OSError as e:
        logging.warning(e)
        dim = PARAMS['wordEmbeddingDim']
        word2vector = util.load_word_embedding("./resources/glove.6B.{}d.txt".format(PARAMS['wordEmbeddingDim']),
                                               dim=dim)
        embedding_matrix = []
        for i in range(len(idx2token)):
            token = idx2token[i]
            vector = word2vector.get(token, np.random.uniform(-0.25, 0.25, dim))
            embedding_matrix.append(vector)

        embedding_matrix = np.asarray(embedding_matrix)

        lstm = build_model(embedding_matrix, PARAMS)
        lstm.fit(X_tr, Y_tr, epochs=5, batch_size=64, validation_data=[X_va, Y_va])
        lstm.save(model_path)

    Y_te_pred_list = lstm.predict(X_te, verbose=1)
    util.submission2(Y_te_pred_list, id_list, output_path="submission2.csv")
