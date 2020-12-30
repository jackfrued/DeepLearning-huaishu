import sys
import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange
from sklearn.metrics import confusion_matrix
from torch.multiprocessing import Pool
from util.Summary import SummaryTool
from util.CustomOptimizer import AdamW
from util.DataSet import trans_text_2_feat
from util.SVMProcessor import SVMProcessor
from util.FastTextProcessor import FastTextProcessor

logger = logging.getLogger(__name__)

class ClsTrainer:
    def __init__(self, model, tokenizer, args, config, device):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = self.args.train_batch_size
        self.n_gpu = torch.cuda.device_count()
        self.config = config
        self.cur_epoch = 0

        self.summary = SummaryTool(self.args.output_dir)

        self.svm_processor = SVMProcessor(self.config,
                                          'tools/postprocess/chi_cate.txt',
                                          'tools/svm/chi_cate_new.txt',
                                          'tools/svm/model_file_c32')
        self.ft_processor = FastTextProcessor('tools/fasttext/ft_classify_model.bin')

        self.result_file_name = {'test': 'preds_results.txt', 
                                 'eval': 'eval_results.txt',
                                 'cls_fig': 'cls_fig.jpg',
                                 'cls_result': 'cls_result.txt'}
        logger.info('Load voc trainer finish')


    def train(self, train_dataLoader, eval_dataSet):
        if self.args.freeze:
            for name, param in self.model.module.named_parameters():
                if 'all' in self.args.freeze or name.startswith(tuple(self.args.freeze)):
                    param.requires_grad = False

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        tr_loss = 0
        global_step = 0
        for epoch in trange(0, self.args.num_train_epochs):
            self.model.train()
            #self.train_dataLoader.sampler.set_epoch(epoch)
            for step, batch in enumerate(tqdm(train_dataLoader, desc='Iteration')):
                step_loss = self.training_step(batch)
                tr_loss += step_loss
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if step % self.summary.record_step == 0:
                    self.summary.add_scalar('train/loss', step_loss, global_step)
            self.cur_epoch += 1

            if self.args.do_eval_after_each_epoch:
                self.eval_epoch(eval_dataSet, epoch)

            if self.args.save_interval > 0 and  (epoch + 1) % self.args.save_interval == 0:
                self.save_model(str(self.cur_epoch))

        self.summary.close_tool()
        self.save_model()


    def training_step(self, batch):
        input_ids, token_type_ids, attention_mask, labels_cls, word_features, cate_word_weights, len_info = batch
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels_cls = labels_cls.to(self.device)
        word_features = word_features.to(self.device)
        cate_word_weights = cate_word_weights.to(self.device)
        len_info = len_info.to(self.device)
        loss_list, logits = self.model(input_ids,
                                  token_type_ids=token_type_ids, 
                                  attention_mask=attention_mask, 
                                  labels_cls=labels_cls, 
                                  word_features=word_features,
                                  cate_word_weights=cate_word_weights,
                                  len_info=len_info,
                                 )
        loss = loss_list
        if self.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.item()

    def eval_epoch(self, dataset, epoch_num):
        score_result_cls, text_info, label_info_cls = self.test_thread(0, dataset)

        if 'cls' not in self.args.freeze:
            matrix_info = np.zeros((self.config.num_labels_cate, self.config.num_labels_cate))
            for a, b in zip(score_result_cls, label_info_cls):
                ps = np.argmax(a)
                ls = np.argmax(b)
                if self.config.originid2label[str(ls)] in self.config.label2id:
                    ls = self.config.label2id[self.config.originid2label[str(ls)]]
                else:
                    ls = self.config.label2id['Tail1>Tail2']
                matrix_info[ls][ps] += 1

            for i in range(self.config.num_labels_cate):
                rowsum = sum(matrix_info[i])
                colsum = sum(matrix_info[r][i] for r in range(self.config.num_labels_cate))
                precision = matrix_info[i][i] / (colsum + 1e-10)
                precision = round(precision, 4)
                recall = matrix_info[i][i] / (rowsum + 1e-10)
                recall = round(recall, 4)
                f1 = round(2*precision*recall/(precision+recall+1e-10), 4)
                cate_name_tmp = self.config.id2label[i]
                self.summary.add_scalars('category/' + cate_name_tmp, 
                                {'precision': precision, 'recall': recall, 'f1': f1},
                                epoch_num)


    def test(self, set_type, test_dataset):
        gpu_num = 8
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].strip().split(','))

        score_all_cls, text_all, label_all_cls = self.test_thread(0, test_dataset)

        pred_label_all_cls = self.unfold_cate(text_all, score_all_cls)

    def test_thread(self, gpu_num, test_data):
        score_result_cls = []
        text_info = []
        label_info_cls = []
        count_num = 0
        self.model.eval()

        for idx, record in enumerate(tqdm(test_data)):
            count_num += 1
            scores = self.test_core([record.text_a, record.text_b], record.word_feat, record.cate_word_weight)
            cls_result_tmp = scores

            if len(record.ft_feat) <= 10:
                ft_result = [0] * self.config.num_labels_cate
                temp_record = '__label__0 ' + ' '.join([str(token) for token in record.ft_feat])
                ft_result_id = self.ft_processor.ft_model.predict(temp_record)[0][0].split('__')[-1]
                ft_result[int(ft_result_id)] = 0.99999999
                cls_result_tmp = ft_result
            score_result_cls.append(cls_result_tmp)

            text_info.append('|||@@@|||'.join([record.text_a, record.text_b]))

            label_vec = [0] * self.config.num_labels_all_cate
            if record.label_cls_all:
                label_vec[self.config.originlabel2id[record.label_cls_all]] = 1
            label_info_cls.append(label_vec)

        return score_result_cls, text_info, label_info_cls


    def test_core(self, str_list, word_feat, cate_word_weight):
        feats = trans_text_2_feat(str_list, self.tokenizer, self.args.max_seq_length)
        feat_input_ids = torch.tensor(feats['input_ids'], dtype=torch.long).to(self.device)
        feat_token_type_ids = torch.tensor(feats['token_type_ids'], dtype=torch.long).to(self.device)
        feat_attention_mask = torch.tensor(feats['attention_mask'], dtype=torch.long).to(self.device)

        word_feat_tensor = [0] * self.config.word_feat_size
        for k in word_feat:
            word_feat_tensor[k] = word_feat[k]
        word_feat_tensor = torch.tensor(word_feat_tensor, dtype=torch.float).to(self.device)
        cate_word_weight = torch.tensor(cate_word_weight, dtype=torch.float).to(self.device)
        len_info = torch.tensor([1.0 * len(str_list[1].split(' '))], dtype=torch.float).to(self.device)

        with torch.no_grad():
            logits = self.model(feat_input_ids.unsqueeze(0),
                                token_type_ids=feat_token_type_ids.unsqueeze(0), 
                                attention_mask=feat_attention_mask.unsqueeze(0), 
                                word_features=word_feat_tensor.unsqueeze(0),
                                cate_word_weights=cate_word_weight.unsqueeze(0),
                                len_info=len_info.unsqueeze(0))
        return logits.detach().cpu().numpy()[0]

    def unfold_cate(self, text_all, score_all_cls):
        ret = []
        for t, s in zip(text_all, score_all_cls):
            ret_temp = {}
            max_idx = np.argmax(s)
            top_k_idx = np.argpartition(s[: -1], -3)[-3:]

            if max_idx != self.config.label2id["Tail1>Tail2"]:
                for tt in top_k_idx:
                    cate = self.config.originlabel2id[self.config.id2label[tt]]
                    score = s[tt]
                    ret_temp[cate] = score
            else:
                text = t.split("|||@@@|||")[1]
                cate_ret = self.svm_processor.get_tail_cate_score_svm(text)
                ret_temp[cate_ret] = 0.9999999
                for tt in top_k_idx[: -1]:
                    cate = self.config.originlabel2id[self.config.id2label[tt]]
                    score = s[tt]
                    ret_temp[cate] = score

            ret_temp = sorted(ret_temp.items(), key=lambda d:d[1], reverse = True)
            ret.append(ret_temp[0][0])

        return ret


    def save_model(self, suffix=None):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        output_model_file = ''
        if suffix is not None:
            output_model_file = os.path.join(self.args.output_dir, 'pytorch_model.bin_' + suffix)
        else:
            output_model_file = os.path.join(self.args.output_dir, 'pytorch_model.bin')
        output_config_file = os.path.join(self.args.output_dir, 'config.json')

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
