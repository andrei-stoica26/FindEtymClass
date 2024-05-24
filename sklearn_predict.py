import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

def run_prediction(words, vectorizer, models, i):
    print(f'Running prediction {i + 1}...')
    words_transformed = vectorizer.transform(words)
    return models[i].predict(words_transformed)

def train_model(X, y, n_state, n_model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = n_state + n_model)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy score for model {n_model}: {accuracy}')
    return model

def sklearn_predict(file_path, column, n_initial_state, new_words_path, n_predictions, pred_column, pred_file):
    df = pd.read_csv(file_path, na_values = ['NA'])
    df = df.replace(pd.NA, 'nan') #Preventing pandas for taking Romanian word 'nan' to mean 'not a number'
    
    vectorizer = TfidfVectorizer(analyzer = 'char', ngram_range = (2, 3))
    X = vectorizer.fit_transform(df['Word'])
    y = df[column]

    models = [train_model(X, y, n_initial_state, n_state + 1) for n_state in range(n_predictions)]

    with open(new_words_path, 'r', encoding = 'utf-8') as f:
        new_words = f.read().split('\n')

    predictions = [run_prediction(new_words, vectorizer, models, i) for i in range(n_predictions)]
    predicted_prob = [sum(x) / n_predictions for x in zip(*predictions)]
    
    df = pd.DataFrame({'Word': new_words, pred_column: predicted_prob})
    df = df.sort_values(by = [pred_column, 'Word'], ascending = [False, True])
    if not os.path.exists('Result'):
        os.makedirs('Result')
    df.to_csv(f'Result/{pred_file}.csv')

def main():
    file_path = 'WikNeo/neo_and_nonneo.csv'
    column = 'isNeologism'
    n_initial_state = 30
    new_words_path = 'Dexonline/Filtered words.txt'
    n_predictions = 100
    pred_column = 'ProbNeologism'
    pred_file = 'sklearn_predictions'
    sklearn_predict(file_path, column, n_initial_state, new_words_path, n_predictions, pred_column, pred_file)

if __name__ == '__main__':
    main()





