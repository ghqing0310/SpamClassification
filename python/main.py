import pickle
import numpy as np
from data import TextData
from train import TextTrain

with open('../data/mailContent_list_1000.pickle', 'rb') as file:
    content_list = pickle.load(file)
    # random.seed(1234)
    # pickle.dump(random.sample(content_list,1000), open('mailContent_list_1000.pickle', 'wb'))
with open('../data/mailLabel_list_1000.pickle', 'rb') as file:
    label_list = pickle.load(file)
    # random.seed(1234)
    # pickle.dump(random.sample(label_list,1000), open('mailLabel_list_1000.pickle', 'wb'))

def statistics():
    # 查看邮件最大长度和平均长度
    length = np.array([len(tmp) for tmp in content_list])
    print(np.max(length)) # 33714
    print(np.mean(length)) # 752
    print(np.median(length)) # 333
    # 统计汉字个数
    print(len(set(''.join(content_list)))) # 9777

config = {
    # train
    'num_iteration': 400,
    'batch_size': 100,
    'learning_rate': 1e-3,
    'print_per_batch': 500 / 10,
    # model
    'embedding_dim': 64,
    'num_filters': 256,
    'kernel_size': 5,
    'hidden_dim': 128,
    'dropout_keep_prob': 0.5
}

td = TextData(content_list, label_list)
train_X, train_Y, test_X, test_Y = td.get_data()
config['vocab_size'] = td.vocab_size
config['num_classes'] = td.num_classes
config['labels'] = td.labels

train = TextTrain(config, train_X, train_Y, test_X, test_Y)
train.cnn_train()
train.cnn_test()
