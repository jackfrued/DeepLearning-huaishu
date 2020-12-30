import nltk

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class Lemmatization:
    def __init__(self, cut_joint_mark = True):
        self.word_net_lemmatizer = WordNetLemmatizer()
        self.cut_joint_mark = cut_joint_mark

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


    def get_lemma(self, str_in):
        str_in = str_in.lower()
        if self.cut_joint_mark:
            str_in = str_in.replace('-', ' ')
        tokens = nltk.word_tokenize(str_in.lower())
        tagged_sent = nltk.pos_tag(tokens)

        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(self.word_net_lemmatizer.lemmatize(tag[0], pos = wordnet_pos))
        return lemmas_sent
