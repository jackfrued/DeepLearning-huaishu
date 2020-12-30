import os

class CateWordInfo:
    def __init__(self, file_path):
        self.buss_cate_2_word = self.load_buss_cate_2_word(file_path)
        self.buss_word_2_cate = self.get_buss_word_2_cate()

    def get_buss_word_2_cate(self):
        buss_word_2_cate = {}
        for cate_key in self.buss_cate_2_word:
            for word_key in self.buss_cate_2_word[cate_key]:
                if word_key not in buss_word_2_cate:
                    buss_word_2_cate[word_key] = []
                buss_word_2_cate[word_key].append(cate_key)
        return buss_word_2_cate

    def load_buss_cate_2_word(self, file_path):
        buss_cate_2_word = {}
        buss_cate_2_word['Running app>App crash/launch error'] = self.load_word_txt(os.path.join(file_path, 'app_crash_lanch_error.txt'))
        buss_cate_2_word['Running app>App installation/update'] = self.load_word_txt(os.path.join(file_path, 'app_installation_update.txt'))
        buss_cate_2_word['Chats'] = self.load_word_txt(os.path.join(file_path, 'chats.txt'))
        buss_cate_2_word['Notification'] = self.load_word_txt(os.path.join(file_path, 'notifications.txt'))
        buss_cate_2_word['Account'] = self.load_word_txt(os.path.join(file_path, 'account.txt'))
        return buss_cate_2_word

    def load_word_txt(self, file_path):
        return dict([(line.strip(), 1) for line in open(file_path)])
