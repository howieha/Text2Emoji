import torch
import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self, NUM_TASKS, BATCH_SIZE, DIM_EMB, VOCAB_SIZE=32000, HIDDEN_DIM=128):
        super(FFNN, self).__init__()
        (self.VOCAB_SIZE, self.BATCH_SIZE, self.DIM_EMB, self.HIDDEN_DIM, self.NUM_TASKS) = (VOCAB_SIZE, BATCH_SIZE, DIM_EMB, HIDDEN_DIM, NUM_TASKS)

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        self.hidden = nn.Linear(self.DIM_EMB, self.HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.HIDDEN_DIM, self.NUM_TASKS)
        self.drop = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, train=False):
        #print(X.size())
        embeds = self.embedding(X)
        out = self.relu(self.hidden(embeds))
        out = torch.sum(out, dim=1)
        out = self.sigmoid(self.drop(self.fc(out)))
        return out