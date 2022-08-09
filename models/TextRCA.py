# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 128                                          # 卷积核数量(channels数)
        self.hidden_size = 128
        self.num_layers = 2


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):  # RNN+ATTENTION 之后与 CNN并联
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)  # batch_first=True表示第一维度是batch_size
        #np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        self.w_omega = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))
        nn.init.normal_(self.w_omega, 0, 1)
        nn.init.normal_(self.u_omega, 0, 1)

        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes) + config.hidden_size * 2, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # 128 * 256 * 31
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 128 * 256
        return x
    def attention(self, out):
        u = torch.tanh(torch.matmul(out, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = out * att_score
        return scored_x

    def forward(self, x):
        # print(x[0].shape)
        out1 = self.embedding(x[0])  # x[0]为128*32，词嵌入之后为128 * 32 * 300
        out1 = out1.unsqueeze(1)  # 128 * 1 * 32 * 300
        out1 = torch.cat([self.conv_and_pool(out1, conv) for conv in self.convs], 1)  # 128 * 768
        out1 = self.dropout(out1)  # 128 * 768
        x, _ = x  # 传入的参数x包括文章对应的索引 + 文章的字数
        out2 = self.embedding(x)  # [batch_size, seq_len, embedding]=[128, 32, 300]
        out2, _ = self.lstm(out2)
        out2 = self.attention(out2)  # 句子最后时刻的 hidden state
        out2 = torch.sum(out2, dim=1)
        out = torch.cat([out1, out2], 1)
        out = self.fc(out)
        return out