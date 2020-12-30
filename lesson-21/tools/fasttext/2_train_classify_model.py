import fasttext

model = fasttext.train_supervised('data_for_ft.txt')
model.save_model('ft_classify_model.bin')
