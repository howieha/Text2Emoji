import torch
import torch.nn as nn
from torch import autograd

class CNN_attention(nn.Module):
    def __init__(self, NUM_TASKS, BATCH_SIZE, DIM_EMB, VOCAB_SIZE=32000, NUM_CHANNELS=128):
        super(CNN_attention, self).__init__()
        (self.VOCAB_SIZE, self.BATCH_SIZE, self.DIM_EMB, self.NUM_CHANNELS, self.NUM_TASKS) = (VOCAB_SIZE, BATCH_SIZE, DIM_EMB, NUM_CHANNELS, NUM_TASKS)

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        self.conv1 =  nn.Conv1d(in_channels=self.DIM_EMB, out_channels=self.NUM_CHANNELS, kernel_size=1)
        self.relu = nn.ReLU()
        self.att = self.init_attention()
        self.fc = nn.Linear(self.NUM_CHANNELS, 1)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def init_attention(self):
        return (autograd.Variable(torch.randn(self.NUM_CHANNELS, self.NUM_TASKS)).cuda())

    def forward(self, X, train=False):
        embedded_sent = self.embedding(X).permute(0, 2, 1)#.unsqueeze(0)
        #print(embedded_sent.size())
        conv_out = self.conv1(embedded_sent).permute(0,2,1)
        att_out = self.softmax(torch.matmul(conv_out, self.att))
        out = torch.bmm(att_out.permute(0,2,1), conv_out)
        out = self.drop(self.fc(out))
        out = self.sigmoid(out.squeeze())
        return out
