import pandas as pd

from transformers import BertConfig

config = BertConfig.from_json_file('../../bert_model_trans/config.json')

fout = open('data_for_ft.txt', 'w')

word_idx_info = {}
word_idx_count = 0
fout_word_idx = open('word_index_info.txt', 'w')

for record in pd.read_csv('./cls_train.csv').values:
    eng = str(record[1]).strip().replace('\n', '').replace('\r', '')
    engs = eng.split(' ')
    engs_pure = []
    for e in engs:
        while e.endswith((',', '.', '?', '!', '-', '<', '>', '(', ')')):
            e = e[0: -1]
        if len(e) > 0:
            if e in word_idx_info:
                engs_pure.append(word_idx_info[e])
            else:
                word_idx_info[e] = str(word_idx_count)
                word_idx_count += 1
                engs_pure.append(word_idx_info[e])
    eng = ' '.join(engs_pure)
    cate = record[-3] + '>' + record[-2]
    cate = str(config.label2id[cate])

    fout.write('__label__' + cate + ' ' + eng + '\n')

fout.close()

for k in word_idx_info:
    fout_word_idx.write(k + '\t' + word_idx_info[k] + '\n')
fout_word_idx.close()
