import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, NUM_TASKS, BATCH_SIZE, DIM_EMB, VOCAB_SIZE=32000, NUM_CHANNELS=128):
        super(CNN, self).__init__()
        (self.VOCAB_SIZE, self.BATCH_SIZE, self.DIM_EMB, self.NUM_CHANNELS, self.NUM_TASKS) = (VOCAB_SIZE, BATCH_SIZE, DIM_EMB, NUM_CHANNELS, NUM_TASKS)
        # TODO: Initialize parameters.
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.DIM_EMB, out_channels=self.NUM_CHANNELS,
                      kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.DIM_EMB, out_channels=NUM_CHANNELS,
                      kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.DIM_EMB, out_channels=NUM_CHANNELS,
                      kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.fc = nn.Linear(2*NUM_CHANNELS, self.NUM_TASKS)
        self.bn = nn.BatchNorm1d(self.NUM_TASKS)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, X, train=False):
        embedded_sent = self.embedding(X).permute(0, 2, 1)#.unsqueeze(0)
        #print(embedded_sent.size())

        conv_out1 = self.conv1(embedded_sent)
        # print(conv_out1.size())
        conv_out2 = self.conv2(embedded_sent)
        # print(conv_out2.size())
        #conv_out3 = self.conv3(embedded_sent)
        # print(conv_out3.size())

        all_out = torch.cat((conv_out1, conv_out2), 1)
        #print(all_out.size())
        all_out = all_out.view(all_out.size(0), -1)
        #print(all_out.size())
        out = self.sigmoid(self.bn(self.fc(all_out)))
        #print(out.size())
        return out
