from tqdm import tqdm
from libsvm.svmutil import *
from libsvm.svm import *

# train 
y, x = svm_read_problem('train_svm.csv')
model = svm_train(y, x, '-h 0 -t 0 -b 1 -e 0.5')
#model = svm_train(y, x, '-h 0 -t 0')
svm_save_model('model_file_c32', model)

# test
yt, xt = svm_read_problem('test_svm.csv')
model = svm_load_model('model_file_c32')
p_labs, p_acc, p_vals = svm_predict(yt, xt, model)
