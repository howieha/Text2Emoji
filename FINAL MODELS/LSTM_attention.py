import torch
import torch.nn as nn
from torch import autograd


class LSTM_attention(nn.Module):
    def __init__(self, NUM_TASKS, BATCH_SIZE, DIM_EMB, VOCAB_SIZE=32000, HIDDEN_DIM=128):
        super(LSTM_attention, self).__init__()
        (self.VOCAB_SIZE, self.BATCH_SIZE, self.DIM_EMB, self.HIDDEN_DIM, self.NUM_TASKS) = (VOCAB_SIZE, BATCH_SIZE, DIM_EMB, HIDDEN_DIM, NUM_TASKS)

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        self.lstm = nn.LSTM(self.DIM_EMB, self.HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.att = self.init_attention()
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(self.HIDDEN_DIM, 1)
        self.drop = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def init_attention(self):
        return (autograd.Variable(torch.randn(self.HIDDEN_DIM, self.NUM_TASKS)).cuda())

    def forward(self, X, train=False):
        #print(X.size())
        embeds = self.embedding(X).permute(1,0,2)
        #print(embeds.size())
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = lstm_out.permute(1,0,2)
        att_out = torch.matmul(lstm_out, self.att)
        att_out = self.softmax(att_out)
        #print(out.size())
        out = torch.bmm(att_out.permute(0, 2, 1), lstm_out)
        out = self.drop(self.fc(out))
        out = self.sigmoid(out.squeeze())
        #print(out.size())
        return out