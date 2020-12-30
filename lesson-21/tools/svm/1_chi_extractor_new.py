#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import pandas as pd
import math

from tqdm import tqdm
from transformers import BertConfig
from Lemmatizer import Lemmatization

config = BertConfig.from_json_file('../../bert_model_trans/config.json')
lemma = Lemmatization()

big_info = ["App rating>No content",
         "App rating>Satisfied",
         "App rating>Dissatisfied",
         "Running app>App crash/launch error",
         "Running app>App installation/update",
         "Chats",
         "Account",
         "Notification",
         "Tail1>Tail2"] 

def deal_word(term):
    term = term.replace(',', '').replace('.', '').replace('?', '').replace(':', '').replace(')', '').replace('(', '').replace('[', '').replace(']', '').replace('"', '').replace('*', '').replace('!', '')
    return term

def key_word(file_path, result_path):
    data_df = pd.read_csv(file_path)

    dic = dict()
    cat_num_dic = dict()
    rows = 0
    word_count = {}

    for (i, record) in enumerate(tqdm(data_df.values)):
        if len(record) < 8:
            continue
        if len(str(record[0])) == 0 or len(str(record[1])) == 0:
            continue
        flag_continue = 0
        for r in record:
            if str(r) == 'nan':
                flag_continue = 1
                break
        if flag_continue == 1:
            continue

        if record[3] + '>' + record[4]  in big_info:
            continue

        content = record[1].strip().lower()
        seg_result = lemma.get_lemma(content)
        cate_label = record[3].strip() + '>' + record[4].strip()

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
    term_score_map = dict()
    for term in dic.keys():
        term = term.strip()
        all_num_term = float(sum(dic[term].values()))
        all_num_cat = float(sum(cat_num_dic.values()))
        chi_score = 0.0
        max_cate = None
        for cat in cat_num_dic.keys():
            A = float(dic[term][cat])
            B = float(cat_num_dic[cat]-A)
            C = float(all_num_term-A)
            D = float(all_num_cat-all_num_term-cat_num_dic[cat]+A)
            if (A+C)*(B+D)*(A+B)*(C+D) == 0:
                chi_score = 0
            else:
                temp_chi = rows*math.pow(A*D-B*C, 2)/((A+C)*(B+D)*(A+B)*(C+D))
                if chi_score < temp_chi:
                    chi_score = temp_chi
                    max_cate = cat
        term_score[term] = chi_score
        term_score_map[term] = max_cate

    sorted_keys = sorted(term_score.items(), key=lambda term_score: term_score[1], reverse=True)
    fout = open(result_path, 'w')
    count_out = 0
    for (i, record) in enumerate(sorted_keys):
        if word_count[record[0]] < 15:
            continue
        if len(record[0]) <= 1:
            continue
        if record[0].isdigit():
            continue
        if term_score_map[record[0]] not in config.originlabel2id:
            continue
        fout.write(record[0] + '\t' + str('%.5f'%math.log(record[1])) + '\t' + term_score_map[record[0]] + '\n')
        if count_out > 2000:
            break
        count_out += 1
    fout.close()

    print(cat_num_dic, len(cat_num_dic), len(sorted_keys))


if __name__ == '__main__':
    key_word('./cls_train.csv', 'chi_cate_new.txt')
