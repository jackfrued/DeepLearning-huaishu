import torch
import os

from torch.multiprocessing import set_start_method
from transformers import BertTokenizer
from transformers import BertConfig
from util.ArgParser import getArgs
from util.Model import ClsModel
from util.DataSet import ClsProcessor
from util.ModelTrainer import ClsTrainer

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def main():
    # Load parameters of training
    args = getArgs()

    # Load model config file
    if not os.path.exists(os.path.join(args.bert_model, 'config.json')):
        raise Exception('{} json file not exists.'.format(os.path.join(args.bert_model, 'config.json')))
    config = BertConfig.from_json_file(os.path.join(args.bert_model, 'config.json'))

    if not args.bert_model:
        raise Exception('Bert model not found. {}'.format(args.bert_model))
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    model = ClsModel.from_pretrained(args.bert_model, config=config)
    if not model:
        raise Exception('Model init failed')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = torch.nn.DataParallel(model)

    processor = ClsProcessor(config, args)

    trainer = ClsTrainer(model, tokenizer, args, config, device)

    eval_dataSet = None
    if args.do_train:
        train_dataLoader = processor.getTrainDataLoader(args.data_dir, tokenizer)
        if args.do_eval or args.do_eval_after_each_epoch:
            eval_dataSet = processor.getDataSet(args.data_dir, 'eval')
        trainer.train(train_dataLoader, eval_dataSet)
        tokenizer.save_vocabulary(args.output_dir)
    if args.do_eval:
        if not eval_dataSet:
            eval_dataSet = processor.getDataSet(args.data_dir, 'eval')
        trainer.test('eval', eval_dataSet)

if __name__ == "__main__":
    main()
