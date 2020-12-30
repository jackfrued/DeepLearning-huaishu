output_dir_voc=outputs-voc
mkdir ${output_dir_voc}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py \
    --data_dir ./voc_data \
    --bert_model ./bert_model_trans \
    --task_name sentvoc-voc \
    --train_batch_size 256 \
    --num_train_epochs 2 \
    --process_num 35 \
    --output_dir ${output_dir_voc} \
    --ft_model_path tools/fasttext/ft_classify_model.bin \
    --ft_model_word_info tools/fasttext/word_index_info.txt \
    --use_chi_feature \
    --use_key_feature \
    --learning_rate 1e-5 \
    --do_train
