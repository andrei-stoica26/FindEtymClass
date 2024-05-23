import make_partitions as mp
import pandas as pd

def main():
    dex_words = mp.read_words_file('Dexonline/All words.txt')
    dex_words = {x for x in dex_words if not 'î' in x[1:-1]}
    wik_words = set(pd.read_csv('WikNeo/neo_and_nonneo.csv')['Word'])
    dex_words = sorted(list(dex_words - wik_words))[1:]#Taking out an empty space pseudo-word
    with open ('Dexonline/Filtered words.txt', 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(dex_words))

if __name__ == '__main__':
    main()
    
    
