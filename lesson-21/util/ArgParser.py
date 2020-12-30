import argparse

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
            help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--bert_model", default=None, type=str, required=True,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
            "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--label_file", default="classes.txt", type=str, required=False,
            help="Show labels info")

    parser.add_argument("--train_batch_size",
            default=32,
            type=int,
            help="Total batch size for training.")

    parser.add_argument('--gradient_accumulation_steps',
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--num_train_epochs",
            default=3,
            type=int,
            help="Total number of training epochs to perform.")

    parser.add_argument("--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.")

    parser.add_argument("--adam_epsilon", 
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.")
    
    parser.add_argument("--freeze",
            type=str,
            nargs='+',
            default=[],
            help="Which layer freeze?")

    parser.add_argument("--task_name",
            default='voc',
            type=str,
            required=True,
            help="The name of the task to train.")

    parser.add_argument("--do_train",
            action='store_true',
            help="Whether to run training.")

    parser.add_argument("--do_eval",
            action='store_true',
            help="Whether to run eval on the eval set.")

    parser.add_argument("--do_test",
            action='store_true',
            help="Whether to run test on the test set.")

    parser.add_argument("--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--process_num",
            type=int,
            default=1,
            help="process_num for the number of gpus used to evaluate")

    parser.add_argument("--use_chi_feature",
            action='store_true',
            help="Chi words will be added")

    parser.add_argument("--use_key_feature",
            action='store_true',
            help="Key words will be added")

    parser.add_argument("--save_interval",
            default=1,
            type=int,
            help="Interval number of training epoch to save model once.")

    parser.add_argument("--do_eval_after_each_epoch",
            action='store_true',
            help="Whether to run eval after each epoch.")

    parser.add_argument("--ft_model_path", default=None, type=str, required=True,help="ft_model_path")

    parser.add_argument("--ft_model_word_info", default=None, type=str, required=True,help="ft_model_word_info")


    args = parser.parse_args()
    return args
