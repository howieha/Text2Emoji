import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, NUM_TASKS, BATCH_SIZE, DIM_EMB, VOCAB_SIZE=32000, HIDDEN_DIM=128):
        super(LSTM, self).__init__()
        (self.VOCAB_SIZE, self.BATCH_SIZE, self.DIM_EMB, self.HIDDEN_DIM, self.NUM_TASKS) = (VOCAB_SIZE, BATCH_SIZE, DIM_EMB, HIDDEN_DIM, NUM_TASKS)

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        self.lstm = nn.LSTM(self.DIM_EMB, self.HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.HIDDEN_DIM, self.NUM_TASKS)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, X, train=False):
        embeds = self.embedding(X)
        #print(embeds.size())
        lstm_out, self.hidden = self.lstm(embeds)
        #print(lstm_out.size())
        #lstm_out = lstm_out.contiguous().view(-1, self.HIDDEN_DIM)
        #print(lstm_out.size())
        out = self.fc(self.relu(lstm_out))
        out = self.sigmoid(out)
        #print(out.size())
        out = out[:, -1]
        #print(out.size())
        return out