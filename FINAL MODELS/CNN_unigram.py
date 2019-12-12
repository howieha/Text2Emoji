import torch
import torch.nn as nn


class CNN_unigram(nn.Module):
    def __init__(self, NUM_TASKS, BATCH_SIZE, DIM_EMB, VOCAB_SIZE=32000, NUM_CHANNELS=128):
        super(CNN_unigram, self).__init__()
        (self.VOCAB_SIZE, self.BATCH_SIZE, self.DIM_EMB, self.NUM_CHANNELS, self.NUM_TASKS) = (VOCAB_SIZE, BATCH_SIZE, DIM_EMB, NUM_CHANNELS, NUM_TASKS)

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.DIM_EMB, out_channels=self.NUM_CHANNELS, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(NUM_CHANNELS, self.NUM_TASKS)
        self.drop = nn.Dropout()
        self.sigmoid = nn.Sigmoid()


    def forward(self, X, train=False):
        embedded_sent = self.embedding(X).permute(0, 2, 1)#.unsqueeze(0)
        #print(embedded_sent.size())
        all_out = self.conv1(embedded_sent)
        # print(all_out.size())
        all_out = all_out.view(all_out.size(0), -1)
        #print(all_out.size())
        out = self.sigmoid(self.drop(self.fc(all_out)))
        #print(out.size())
        return out
