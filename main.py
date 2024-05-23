import os
import scrape_categories as sc
import make_partitions as mp
import filter_dexonline_words as fdw
import predict_neologisms as pn

def main():
    if not os.path.exists('Categories'):
        sc.main()
    mp.main()
    fdw.main()
    pn.main()

if __name__ == '__main__':
    main()
    
    
