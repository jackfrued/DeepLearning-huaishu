#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import pandas as pd
import math
import csv

from tqdm import tqdm
from transformers import BertConfig

config = BertConfig.from_json_file('../../bert_model_trans/config.json')

idx = 0
word_info = {}
for line in open('./chi_cate_new.txt'):
    word = line.split('\t')[0].strip()
    if word not in word_info:
        word_info[word] = idx
        idx += 1

count_big = 0
count_short = 0
count_all = 0
fout = open('test_svm.csv', 'w')
for record in csv.reader(open('./cls_test.csv')):
    count_all += 1
    contents = record[1].strip().lower().split(' ')

    if record[3] + '>' + record[4] in config.label2id:
        count_big += 1
        continue
    temp_str = str(config.originlabel2id[record[3] + '>' + record[4]])

    temp_info = {}
    seg_result = [w.replace('.', '').replace('?', '').replace('!', '').replace(',', '') for w in contents]
    for s in seg_result:
        if s in word_info:
            temp_info[word_info[s]]= 1
    if len(temp_info) < 1:
        count_short += 1
        continue

    temp_info=sorted(temp_info.items(),key=lambda x:x[0])

    sample = []
    for t in temp_info:
        k, v = t
        sample.append(str(k) + ':' + str(v))
    fout.write(temp_str + ' ' + ' ' .join(sample) + '\n')
fout.close()
print(count_big, count_short, count_all)
