import pandas as pd

from transformers import BertConfig

config = BertConfig.from_json_file('../../bert_model_trans/config.json')

fout = open('data_for_ft_test.txt', 'w')

word_idx_info = {}
for line in open('word_index_info.txt'):
    w, idx = line.strip().split('\t')
    word_idx_info[w] = idx

for record in pd.read_csv('./cls_test.csv').values:
    eng = str(record[1]).strip().replace('\n', '').replace('\r', '')
    engs = eng.split(' ')
    engs_pure = []
    for e in engs:
        while e.endswith((',', '.', '?', '!', '-', '<', '>', '(', ')')):
            e = e[0: -1]
        if len(e) > 0:
            if e in word_idx_info:
                engs_pure.append(word_idx_info[e])
    eng = ' '.join(engs_pure)
    cate = record[-3] + '>' + record[-2]
    if cate in config.label2id:
        cate = str(config.label2id[cate])
        fout.write('__label__' + cate + ' ' + eng + '\n')
    else:
        fout.write('__label__8 ' + cate + ' ' + eng + '\n')

fout.close()
