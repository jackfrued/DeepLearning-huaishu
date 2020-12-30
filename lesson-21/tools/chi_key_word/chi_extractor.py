#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import pandas as pd
import math

from tqdm import tqdm

def deal_word(term):
    term = term.replace(',', '').replace('.', '').replace('?', '').replace(':', '').replace(')', '').replace('(', '').replace('[', '').replace(']', '').replace('"', '').replace('*', '').replace('!', '')
    return term


def key_word(file_path, result_path, skip_flag=False):
    data_df = pd.read_csv(file_path)

    dic = dict()
    cat_num_dic = dict()
    rows = 0
    word_count = {}

    for (i, record) in enumerate(tqdm(data_df.values)):
        if len(record) != 6:
            continue

        content = record[1]
        seg_result = [deal_word(w) for w in str(record[1]).split(' ')]
        cate_label = record[-2] + '>' + record[-1]

        rows += 1
        if cate_label in cat_num_dic:
            cat_num_dic[cate_label] += 1.0
        else:
            cat_num_dic[cate_label] = 1.0

        temp_dict = dict()
        for i in range(0, len(seg_result)):
            term = seg_result[i].strip()
            if len(term) == 0:
                continue

            temp_dict[term] = 1.0
            if term not in word_count:
                word_count[term] = 1
            else:
                word_count[term] += 1

            if not skip_flag:
                continue

            if i + 1 < len(seg_result):
                term = ''.join([t.strip() for t in seg_result[i: i + 2]])
                if term not in word_count:
                    word_count[term] = 1
                else:
                    word_count[term] += 1

            if i + 2 < len(seg_result):
                term = ''.join([t.strip() for t in seg_result[i: i + 3]])
                if term not in word_count:
                    word_count[term] = 1
                else:
                    word_count[term] += 1

            if i + 3 < len(seg_result):
                term = ''.join([t.strip() for t in seg_result[i: i + 4]])
                if term not in word_count:
                    word_count[term] = 1
                else:
                    word_count[term] += 1

        for key in temp_dict:
            if key in dic:
                if cate_label in dic[key]:
                    dic[key][cate_label] += 1.0
                else:
                    dic[key][cate_label] = 1
            else:
                dic[key] = {}
                dic[key][cate_label] = 1

    for key in dic.keys():
        for cat in cat_num_dic.keys():
            if cat not in dic[key]:
                dic[key][cat] = 0.0

    term_score = dict()
    for term in dic.keys():
        term = term.strip()
        all_num_term = float(sum(dic[term].values()))
        all_num_cat = float(sum(cat_num_dic.values()))
        chi_score = 0.0
        for cat in cat_num_dic.keys():
            if cat == 'AT90' or cat == "AT40":
                continue
            A = float(dic[term][cat])
            B = float(cat_num_dic[cat]-A)
            C = float(all_num_term-A)
            D = float(all_num_cat-all_num_term-cat_num_dic[cat]+A)
            if (A+C)*(B+D)*(A+B)*(C+D) == 0:
                chi_score = 0
            else:
                chi_score = max(chi_score, rows*math.pow(A*D-B*C, 2)/((A+C)*(B+D)*(A+B)*(C+D)))
        term_score[term] = chi_score

    sorted_keys = sorted(term_score.items(), key=lambda term_score: term_score[1], reverse=True)
    count = 0
    fout = open(result_path, 'w')
    print(len(sorted_keys))
    for (i, record) in enumerate(sorted_keys):
        if count > 2000:
            break
        if word_count[record[0]] < 15:
            continue
        if len(record[0]) <= 1:
            continue
        if record[0].isdigit():
            continue
        fout.write(record[0] + '\t' + str(count) + '\n')
        count += 1
    fout.close()


if __name__ == '__main__':
    key_word('/workspace/data-2.0/make_balance_data/train_set/voc_train.csv', 'chi_word_not_skip.txt', False)
