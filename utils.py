import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import os

#Reading labelled words and unlabelled (new) words
def read_data(words_path, labels_col, new_words_path):
    df = pd.read_csv(words_path, na_values = ['NA'])
    df = df.replace(pd.NA, 'nan') #Preventing pandas for taking Romanian word 'nan' to mean 'not a number'
    return df['Word'].to_numpy(), df[labels_col].to_numpy(), Path(new_words_path).read_text(encoding = 'utf-8').split('\n')

def char_ngrams_vectorize(words):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    X = vectorizer.fit_transform(words)
    return X, vectorizer

def run_prediction(new_words_transformed, models, i, lib, batch_size):
    print(f'Running prediction {i + 1}...')
    if lib == 'sklearn':
        return models[i].predict(new_words_transformed)
    if lib == 'tf':
        return models[i].predict(new_words_transformed, batch_size).squeeze()

def average_predictions(new_words_transformed, models, lib, batch_size=0):
    predictions = [run_prediction(new_words_transformed, models, i, lib, batch_size) for i in range(len(models))]
    return [sum(x) / len(models) for x in zip(*predictions)]

def write_results(pred_file, new_words, pred_column, pred_prob):
    df = pd.DataFrame({'Word': new_words, pred_column: pred_prob})
    df = df.sort_values(by = [pred_column, 'Word'], ascending = [False, True])
    if not os.path.exists('Result'):
        os.makedirs('Result')
    df.to_csv(f'Result/{pred_file}.csv')
