from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflow.keras as kr
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np

class TextData():
    def __init__(self, *args):
        content_list = args[0]
        label_list = args[1]
        train_X, test_X, train_y, test_y = train_test_split(content_list, label_list, random_state=1234)
        self.train_content_list = train_X
        self.train_label_list = train_y
        self.test_content_list = test_X
        self.test_label_list = test_y
        self.content_list = self.train_content_list + self.test_content_list
        self.num_classes = np.unique(self.train_label_list + self.test_label_list).shape[0]
        self.embedding_dim = 64 # 词向量维度
        self.seq_length = 1000

    def __prepare_data(self):
        # 获取词汇表，更新词汇大小和内容长度
        counter = Counter(''.join(self.content_list))
        vocabulary_list = ['PAD'] + [k[0] for k in counter.most_common()] # 按出现次数倒序排列
        self.vocab_size = len(vocabulary_list)
        # 构建词-id的映射，将label转为0和1
        self.word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.train_label_list) #将ham和spam转为0和1
        self.labels = self.label_encoder.classes_

    # 给每封邮件添加padding（在前）
    def __content2X(self, content_list):
        idlist_list = [[self.word2id_dict[word] for word in content if word in self.word2id_dict] for content in content_list]
        X = kr.preprocessing.sequence.pad_sequences(idlist_list, self.seq_length) 
        return X

    def __label2Y(self, label_list):
        return self.label_encoder.transform(label_list)

    def get_data(self):
        self.__prepare_data()
        return (self.__content2X(self.train_content_list), self.__label2Y(self.train_label_list), 
                self.__content2X(self.test_content_list), self.__label2Y(self.test_label_list))