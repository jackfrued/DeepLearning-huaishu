import os
import torch
import logging

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class ClsModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        logger.info('Finish loading bert base model: ')

        # For classify
        self.num_labels_cate = config.num_labels_cate
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_fc1= nn.Linear(config.hidden_size * 2 + config.word_feat_size + config.num_labels_cate, config.hidden_size)
        self.cls_fc2 = nn.Linear(config.hidden_size, config.num_labels_cate)
        self.activation = nn.Tanh()

        self.init_weights()
        self.voc_max_seq_length = 128
        logger.info('Load voc model finish')
    
    def forward(self, 
                input_ids,
                token_type_ids=None, 
                attention_mask=None, 
                labels_cls=None, 
                word_features=None,
                cate_word_weights=None,
                len_info=None):
        tensor_cat = []
        for i in range(2):
            outputs = self.bert(
                    input_ids[:, i*self.voc_max_seq_length:(i+1)*self.voc_max_seq_length],
                    attention_mask=attention_mask[:, i*self.voc_max_seq_length:(i+1)*self.voc_max_seq_length],
                    token_type_ids=token_type_ids[:, i*self.voc_max_seq_length:(i+1)*self.voc_max_seq_length],
                    )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            tensor_cat.append(pooled_output)

        # classify net
        bert_word_feat = torch.cat(tensor_cat + [word_features] + [cate_word_weights], 1)

        t_out = self.activation(self.cls_fc1(bert_word_feat))
        t_logits = self.cls_fc2(self.dropout(t_out))
        logits = nn.functional.softmax(t_logits, dim=1)
        
        loss_cls = torch.tensor(0.0).cuda()
        if labels_cls is not None:
            weight = 0.8 + len_info * 0.2 / 100
            loss_fct = BCEWithLogitsLoss(weight = weight)
            loss_cls = loss_fct(logits.view(-1, self.num_labels_cate), labels_cls.view(-1, self.num_labels_cate).float())
            return loss_cls, logits
        else:
            return logits
