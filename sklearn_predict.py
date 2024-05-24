from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import utils

def train_model(X, y, n_state, n_model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = n_state + n_model)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy score for model {n_model}: {accuracy}')
    return model

def sklearn_predict(words_path, new_words_path, labels_col, n_initial_state, n_predictions, pred_column, pred_file):
    words, y, new_words = utils.read_data(words_path, labels_col, new_words_path)
    X, vectorizer = utils.char_ngrams_vectorize(words)
    models = [train_model(X, y, n_initial_state, i) for i in range(n_predictions)]
    new_words_transformed = vectorizer.transform(new_words)
    pred_prob = utils.average_predictions(new_words_transformed, models, 'sklearn')
    utils.write_results(pred_file, new_words, pred_column, pred_prob)

def main():
    words_path = 'WikNeo/neo_and_nonneo.csv'
    new_words_path = 'Dexonline/Filtered words.txt'
    labels_col = 'isNeologism'
    n_initial_state = 30
    n_predictions = 100
    pred_column = 'ProbNeologism'
    pred_file = 'sklearn_predictions'
    sklearn_predict(words_path, new_words_path, labels_col, n_initial_state, n_predictions, pred_column, pred_file)
    print('Finished. Continuing with the TensorFlow model.')

if __name__ == '__main__':
    main()





