import os
import scrape_categories as sc
import make_partitions as mp
import filter_dexonline_words as fdw
import sklearn_predict as sp
import tensorflow_predict as tp

def main():
    if not os.path.exists('Categories'):
        sc.main()
    mp.main()
    fdw.main()
    sp.main()

if __name__ == '__main__':
    main()
    
    
