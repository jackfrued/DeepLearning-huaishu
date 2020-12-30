from libsvm.svmutil import *
from libsvm.svm import *

class SVMProcessor:
    def __init__(self, config, tail_word_file_path, svm_word_file_path, svm_model_path):
        self.config = config
        self.word_cate_dict = self.load_chi_info(tail_word_file_path)
        self.word_info = self.load_svm_info(svm_word_file_path)
        self.svm_model = svm_load_model(svm_model_path)

    def load_svm_info(self, svm_word_file_path):
        temp_word_info = {}
        idx = 0
        for line in open(svm_word_file_path):
            word = line.split('\t')[0].strip()
            if word not in temp_word_info:
                temp_word_info[word] = idx
                idx += 1
        return temp_word_info

    def load_chi_info(self, tail_word_file_path):
        word_cate_dict = {}
        for record in open(tail_word_file_path):
            word, score, cate_label = record.strip().split('\t')
            if word not in word_cate_dict:
                word_cate_dict[word] = []
            word_cate_dict[word] = [float(score), cate_label]
        return word_cate_dict

    def get_tail_cate_score(self, str_in):
        content_words = self.remove_punc_and_split(str_in)
        cate_result = {}
        for word in content_words:
            if word in self.word_cate_dict:
                score, cate_label = self.word_cate_dict[word]
                if cate_label not in cate_result:
                    cate_result[cate_label] = 0.0
                cate_result[cate_label] += score

        if len(cate_result) == 0:
            return 0 # No content category

        max_cate = 0
        max_score = 0
        for c in cate_result:
            if cate_result[c] > max_score:
                max_score = cate_result[c]
                max_cate = c
        return int(self.config.originlabel2id[max_cate])

    def get_tail_cate_score_svm(self, str_in):
        content_words = self.remove_punc_and_split(str_in)
        temp_dict = {}
        for c in content_words:
            if c in self.word_info:
                temp_dict[self.word_info[c]] = 1
        if len(temp_dict) < 1:
            return 0
            #return self.get_tail_cate_score(str_in)

        label, _, _ = svm_predict([0], [temp_dict], self.svm_model, '-q')
        return int(label[0])


    """
        Remove punctuations and split the string

        @param string str_in Input sting

        @return list Returns the processed string
    """
    def remove_punc_and_split(self, str_in):
        words = [w.replace('.', '').replace('?', '').replace('!', '').replace(',', '') for w in str_in.lower().strip().split(' ') if w]
        return words
