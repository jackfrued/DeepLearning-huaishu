import fasttext
import torch

class FastTextProcessor:
    def __init__(self, model_path):
        self.ft_model = fasttext.load_model(model_path)

    def get_sentence_vec(self, sentence_ids):
        end_pos = list(sentence_ids).index(-1)
        sentence = ' '.join([str(i) for i in sentence_ids[0: end_pos]])
        return self.ft_model.get_sentence_vector(sentence)

    def get_sentence_vec_batch(self, batch_data):
        vecs = []
        for b in batch_data:
            vecs.append(self.get_sentence_vec(b))
        return torch.tensor(vecs, dtype=torch.float)

    def get_sentence_vec_simple(self, sentence_ids):
        sentence = ' '.join([str(i) for i in sentence_ids])
        return self.ft_model.get_sentence_vector(sentence)
