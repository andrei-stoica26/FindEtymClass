import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import utils

def train_model(X, y, tokenizer, n_state, n_model, batch_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = n_state + n_model)
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f'Fitting model {n_model + 1}...')
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=batch_size, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    return model

def tensorflow_predict(words_path, new_words_path, labels_col, n_initial_state, batch_size, n_predictions, pred_column, pred_file):
    words, y, new_words = utils.read_data(words_path, labels_col, new_words_path)
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(words)
    sequences = tokenizer.texts_to_sequences(words)
    max_length = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, maxlen = max_length, padding = 'post')
    models = [train_model(X, y, tokenizer, n_initial_state, i, batch_size) for i in range(n_predictions)]
    new_sequences = tokenizer.texts_to_sequences(new_words)
    new_padded = pad_sequences(new_sequences, maxlen = max_length, padding='post')
    pred_prob = utils.average_predictions(new_padded, models, 'tf')
    utils.write_results(pred_file, new_words, pred_column, pred_prob)

def main():
    with open('tf_output.log', 'w', encoding = 'utf-8') as f:
        sys.stdout = f
        words_path = 'WikNeo/neo_and_nonneo.csv'
        new_words_path = 'Dexonline/Filtered words.txt'
        labels_col = 'isNeologism'
        n_initial_state = 30
        n_predictions = 1
        batch_size = 32
        pred_column = 'ProbNeologism'
        pred_file = 'tensorflow_predictions'
        tensorflow_predict(words_path, new_words_path, labels_col, n_initial_state, batch_size, n_predictions, pred_column, pred_file)

if __name__ == '__main__':
    main()
