import pandas as pd
import os
import math

def add_file_names(categories):
    return ['Categories/wik_' + s + '.txt' for s in categories]

def read_words_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.read().split('\n')
    return {x for x in words if not set(x) - set('aăâbcdefghiîjlmnoprsștțuvxz')}

def word_lists_union(files):
    word_lists = [read_words_file(x) for x in files]
    union_set = set()
    for x in word_lists:
        union_set = union_set.union(x) 
    return union_set

def make_partitions(part1_tags, part2_tags, column, folder_name, file_name):
    part1_files = add_file_names(part1_tags)
    part2_files = add_file_names(part2_tags)
    part1 = word_lists_union(part1_files)
    part2 = word_lists_union(part2_files)
    part1, part2 = sorted(list(part1 - part2)), sorted(list(part2 - part1))
    df = pd.DataFrame({'Word': part1 + part2, column: [1] * len(part1) + [0] * len(part2)})
    df = df.sort_values(by = 'Word')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    df.to_csv(f'{folder_name}/{file_name}.csv', encoding = 'utf-8')
    

def main():
    part1_tags = ['french', 'latin_neo', 'italian', 'english']
    part2_tags = ['latin', 'hungarian', 'ocs', 'unknown', 'ottoman',  'greek',  'ukrainian', 'bulgarian']
    column = 'isNeologism'
    folder_name = 'WikNeo'
    file_name = 'neo_and_nonneo'
    make_partitions(part1_tags, part2_tags, column, folder_name, file_name)

if __name__ == '__main__':
    main()



