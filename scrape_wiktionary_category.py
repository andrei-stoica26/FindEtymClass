import requests
import os
from bs4 import BeautifulSoup
import time
from fpdf import FPDF

def get_words_from_page(soup):
    links = soup.find_all('a')
    filters = ['Appendix', 'Categor', 'Help', 'Special', 'Wiktionary', 'wiktionary.org'] #The last element filters out Wiktionary pages in other languages
    words = [x.get_text() for x in links if x.get('href') and '/wiki/' in x.get('href') and all ([y not in x.get('href') for y in filters])]
    return words

def scrape_wiktionary_category(url, folder_name, file_title, do_save_as_pdf = True):
    words = []
    page_number = 1
    category = url.split('/')[-1]
    while True:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        words.extend(get_words_from_page(soup))
        print(f'Scraped page {page_number} from {category}.')
        page_number += 1
        next_page_link = soup.find('a', string = 'next page')
        if not next_page_link:
            print(f'Completed scraping {category}.')
            break 
        url = 'https://en.wiktionary.org' + next_page_link.get('href')
        time.sleep(2)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(f'{folder_name}/{file_title}.txt', 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(sorted(list(set(words)))))
        print(f'Saved file: {folder_name}/{file_title}.txt')
    if do_save_as_pdf:
        pdf = FPDF()   
        pdf.add_page()
        pdf.set_font('Times', size = 12)
        with open(f'{folder_name}/{file_title}.txt') as f:
            for x in f:
                pdf.cell(200, 10, txt = x, ln = 1)
        pdf.output(f'{folder_name}/{file_title}.pdf')
        print(f'Saved file: {folder_name}/{file_title}.pdf')

def main():
    url = 'https://en.wiktionary.org/wiki/Category:Galician_terms_with_unknown_etymologies'
    folder_name = 'Output'
    file_title = 'words'
    scrape_wiktionary_category(url, folder_name, file_title, True)

if __name__ == '__main__':
    main()




