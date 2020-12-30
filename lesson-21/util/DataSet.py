import os
import sys
import pandas as pd
import torch
import logging

from tqdm import tqdm
from transformers import InputExample
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
from util.AdditionalFeatures import *
from util.Lemmatizer import Lemmatization
from util.CateInfo import CateWordInfo

logger = logging.getLogger(__name__)

class InputExample:
    def __init__(self, 
                 guid, 
                 text_a=None, 
                 text_b=None, 
                 label_cls=None,
                 word_feat=None,
                 ft_feat=None,
                 lemma_eng=None,
                 cate_word_weight=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_cls = label_cls
        self.ft_feat = ft_feat
        self.word_feat=word_feat
        self.lemma_eng=lemma_eng
        self.cate_word_weight=cate_word_weight

class ClsProcessor:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.ft_word_info = {}
        self.lemma = Lemmatization()
        self.cate_word_info = CateWordInfo('tools/postprocess/buss_word')
        for line in open(self.args.ft_model_word_info):
            w, idx = line.strip().split('\t')
            self.ft_word_info[w] = int(idx)

        if self.args.use_chi_feature and self.args.use_key_feature:
            self.word_feature = WordFeature('tools/chi_key_word/chi_key_word.txt')

    def getDataSet(self, data_path, set_type:str, data_file=None):
        data_set = self._get_examples(os.path.join(data_path, data_file if data_file else 'cls_test.csv'), set_type)
        return data_set

    def getTrainDataLoader(self, data_path, tokenizer, train_file=None):
        train_set = self._get_examples(os.path.join(data_path, train_file if train_file else 'cls_train.csv'), 'train')

        train_dataLoader = self._get_dataloader(train_set, 'train', tokenizer)

        return train_dataLoader

    def _get_examples(self, data_path, set_type:str):
        lines = pd.read_csv(data_path, header=0, encoding='utf-8')
        lines = list(lines.values)
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            if i > 1000:
                break
            bad_flag = 0
            for l in line:
                if str(l).lower() == 'nan':
                    bad_flag = 1
                    break
            if bad_flag == 1:
                continue

            if line[5].strip() + '>' + line[6].strip() not in self.config.label2id:
                continue

            guid = '%s-%s' % (set_type, i)
            text_a = line[0].strip().replace('\r', '').replace('\n', '')
            text_b = line[1].strip().replace('\r', '').replace('\n', '')
            if len(text_a) < 1 or len(text_b) < 1:
                continue

            label_cls = None if set_type != 'train' else line[-3].strip() + '>' + line[-2].strip()
            
            word_feat = self.word_feature.get_key_chi_vec(text_b)

            ft_feat = []
            for word in text_b.split(' '):
                while word.endswith((',', '.', '?', '!', '-', '<', '>', '(', ')')):
                    word = word[0: -1]
                if len(word) > 0:
                    if word in self.ft_word_info:
                        ft_feat.append(self.ft_word_info[word])
            if len(ft_feat) > self.args.max_seq_length:
                ft_feat = ft_feat[0: self.args.max_seq_length]

            lemma_eng = self.lemma.get_lemma(text_b)
            cate_word_weight = [0] * self.config.num_labels_cate
            for token in lemma_eng:
                if token in self.cate_word_info.buss_word_2_cate:
                    for cate_temp in self.cate_word_info.buss_word_2_cate[token]:
                        if cate_temp not in self.config.label2id:
                            cate_word_weight[self.config.label2id['Tail1>Tail2']] += 1
                        else:
                            cate_word_weight[self.config.label2id[cate_temp]] += 1

            examples.append(InputExample(guid=guid, 
                                        text_a=text_a, text_b=text_b, 
                                        label_cls=label_cls, 
                                        word_feat=word_feat, 
                                        ft_feat=ft_feat,
                                        lemma_eng=lemma_eng,
                                        cate_word_weight=cate_word_weight))
        print(set_type, 'example num is: ', len(examples))
        return examples


    def _get_dataloader(self, examples, set_type, tokenizer):
        if not examples:
            raise ValueError(set_type + ' data is empty')

        feat_input_ids = []
        feat_token_type_ids = []
        feat_attention_mask =[]
        labels_cls = []
        word_feats = []
        cate_word_weights = []
        len_info = []
        count_num = 0
        for record in examples:
            if not isinstance(record.text_a, str) or not isinstance(record.text_b, str):
                continue
            sys.stdout.write('processing: ' + str(count_num) + '\r')
            sys.stdout.flush()
            count_num += 1
            feats = trans_text_2_feat([record.text_a, record.text_b], tokenizer, self.args.max_seq_length)
            feat_input_ids.append(feats['input_ids'])
            feat_token_type_ids.append(feats['token_type_ids'])
            feat_attention_mask.append(feats['attention_mask'])
            label_vec = [0] * self.config.num_labels
            if record.label_cls in self.config.label2id:
                label_vec[self.config.label2id[record.label_cls]] = 1
            labels_cls.append(label_vec)

            temp_feat = [0] * self.word_feature.keyword_num
            for k in record.word_feat:
                temp_feat[k] = record.word_feat[k]
            word_feats.append(temp_feat)
            cate_word_weights.append(record.cate_word_weight)
            len_info.append([1.0 * len(record.text_b.split(' '))])

        feat_input_ids = torch.tensor(feat_input_ids, dtype=torch.long)
        feat_token_type_ids = torch.tensor(feat_token_type_ids, dtype=torch.long)
        feat_attention_mask = torch.tensor(feat_attention_mask, dtype=torch.long)
        labels_cls = torch.tensor(labels_cls, dtype=torch.long)

        word_feats = torch.tensor(word_feats, dtype=torch.float)
        cate_word_weights = torch.tensor(cate_word_weights, dtype=torch.float)
        len_info = torch.tensor(len_info, dtype=torch.float)

        comb_data = TensorDataset(feat_input_ids, feat_token_type_ids, feat_attention_mask, 
                                  labels_cls, word_feats, cate_word_weights, len_info) 
        data_loader = DataLoader(
            comb_data,
            batch_size=self.args.train_batch_size,
            sampler=RandomSampler(comb_data),
            drop_last=True,
        )

        return data_loader


def trans_text_2_feat(str_list, tokenizer, max_seq_length):
    ret = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
    for s in str_list:
        tokens = tokenizer(s, add_special_tokens=False)
        if len(tokens['input_ids']) > max_seq_length - 2:
            tokens['input_ids'] = tokens['input_ids'][: max_seq_length - 2]
        need_len = max_seq_length - 2 - len(tokens['input_ids'])

        input_ids = [101] + tokens['input_ids'] + [102] + [0] * need_len
        token_type_ids = [0] * max_seq_length
        attention_mask = [1] * max_seq_length
            
        ret['input_ids'].extend(input_ids)
        ret['token_type_ids'].extend(token_type_ids)
        ret['attention_mask'].extend(attention_mask)
    
    return ret


