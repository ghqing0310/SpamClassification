import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_filters, kernel_size, hidden_dim, dropout_keep_prob, num_classes):
        super(CNN,self).__init__()
        
        self.embeds = nn.Embedding(vocab_size, embedding_dim) # 随机词向量

        self.conv = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=num_filters, # 通道数即过滤器个数
                              kernel_size=kernel_size)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dense = nn.Linear(num_filters, hidden_dim)
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embeds(x) # train_size*seq_length*embedding_dim
        x = x.permute(0,2,1) # train_size*embedding_dim*seq_length
        x = self.conv(x) # train_size*num_filters*conv(seq_length)
        x = self.maxpool(x)[:,:,0] # train_size*num_filters
        x = self.dense(x) # train_size*hidden_dim
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense2(x) # train_size*num_classes
        return x