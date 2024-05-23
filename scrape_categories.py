import scrape_wiktionary_category as swc

def main():
    urls = ['https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_French',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Latin',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Italian',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_English',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_German',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Russian',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_inherited_from_Latin',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Hungarian',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Old_Church_Slavonic',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_with_unknown_etymologies',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Ottoman_Turkish',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Greek',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Ukrainian',
            'https://en.wiktionary.org/wiki/Category:Romanian_terms_borrowed_from_Bulgarian',
            ]
    titles = ['wik_' + s for s in ['french', 'latin_neo', 'italian', 'english', 'german', 'russian', 'latin', 'hungarian', 'ocs', 'unknown', 'ottoman',  'greek',  'ukrainian', 'bulgarian']]
    folder_name = 'Categories'
    for x, y in zip(urls, titles):
        swc.scrape_wiktionary_category(x, folder_name, y, False)

if __name__ == '__main__':
    main()




