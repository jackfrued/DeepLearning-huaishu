import sys
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

class WordFeature:
    def __init__(self, word_file):
        self.keyword_dict = {a: int(b.strip()) for a,b in [line.split('\t') for line in open(word_file,'r',encoding='utf-8')]}
        self.keyword_num = len(self.keyword_dict)

    def get_key_chi_vec(self, str_in):
        strs = str_in.split(' ')
        ret = {}
        for s in strs:
            if s in self.keyword_dict:
                if s in ret:
                    ret[self.keyword_dict[s]] += 1
                else:
                    ret[self.keyword_dict[s]] = 1
        return ret

class RateFeature:
    def get_feature(self, str_in):
        return int(str_in)
