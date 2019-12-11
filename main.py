import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as d
from sklearn import metrics

import utils
from model.LSTM import LSTM
from model.CNN import CNN
from dataset import EmojiDataset

BATCH_SIZE = 128
DIM_EMB = 128
NUM_TASKS = 50
N_EPOCH = 10
LEARNING_RATE = 1e-3
DATA_FILE = r"./data/train_succinct50.pickle"

def Pred(test_loader, model):
    preds = []
    y_true = []
    for x, y in test_loader:
        x = x.cuda()
        probs = model.forward(x, train=False).cuda()
        pred = torch.round(probs).cpu().detach().tolist()
        preds.extend(pred)
        y_true.extend(y.tolist())
    return np.array(preds), np.array(y_true)

def Test(test_loader, model):
    filename = sys.argv[1]+"_"+sys.argv[2]+"_result.txt"
    f = open(filename, "w+")
    acc_scores = []
    pre_scores = []
    rec_scores = []
    f1_scores = []
    roc_scores = []
    preds, y_true = Pred(test_loader, model)
    for i in range(preds.shape[1]):
        pred = preds[:, i].astype(int)
        y = y_true[:, i].astype(int)
        #pred_sum = np.sum(pred)
        #if pred_sum > 0:
            #print(i)
            #print(pred_sum)
        acc_score = metrics.accuracy_score(y, pred)
        pre_score = metrics.precision_score(y, pred)
        rec_score = metrics.recall_score(y, pred)
        f1_score = metrics.f1_score(y, pred)
        roc_score = metrics.roc_auc_score(y, pred)
        acc_scores.append(acc_score)
        pre_scores.append(pre_score)
        rec_scores.append(rec_score)
        f1_scores.append(f1_score)
        roc_scores.append(roc_score)
    print("Accuracy: %s" % (np.mean(acc_scores)))
    print("Precision: %s" % (np.mean(pre_scores)))
    print("Recall: %s" % (np.mean(rec_scores)))
    print("F1: %s" % (np.mean(f1_scores)))
    print("ROC-AUC: %s" % (np.mean(roc_scores)))
    f.write("Accuracy scores: " + str(np.round(acc_scores, 3)) + "\n" + str(np.mean(acc_scores)) +"\n")
    f.write("Precision scores: " + str(np.round(pre_scores, 3)) + "\n" + str(np.mean(pre_scores)) +"\n")
    f.write("Recall scores: " + str(np.round(rec_scores, 3)) + "\n" + str(np.mean(rec_scores)) +"\n")
    f.write("F1 scores: " + str(np.round(f1_scores, 3)) + "\n" + str(np.mean(f1_scores)) +"\n")
    f.write("ROC-AUC scores: " + str(np.round(roc_scores, 3)) + "\n" + str(np.mean(roc_scores)) +"\n")
    f.close()

def Val(val_loader, model):
    roc_scores = []
    preds, y_true = Pred(val_loader, model)
    for i in range(preds.shape[1]):
        pred = preds[:, i].astype(int)
        y = y_true[:, i].astype(int)
        roc_score = metrics.roc_auc_score(y, pred)
        roc_scores.append(roc_score)
    return np.mean(roc_scores)

def Train(train_loader, val_loader, weight_pos):
    print("Start Training!")
    if sys.argv[1] == "LSTM":
        model = LSTM(NUM_TASKS, BATCH_SIZE, DIM_EMB).cuda()
    elif sys.argv[1] == "CNN":
        model = CNN(NUM_TASKS, BATCH_SIZE, DIM_EMB).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.BCEWithLogitsLoss(pos_weight=weight_pos.cuda())
    last_val_score = 0.0
    for epoch in range(N_EPOCH):
        print("epoch " + str(epoch)+": ")
        total_loss = 0.0
        #i = 0
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            model.zero_grad()
            probs = model.forward(x).cuda()
            #print(i)
            #print(probs)
            loss = loss_criterion(probs, y)
            total_loss += loss
            loss.backward()#retain_graph=True)
            optimizer.step()
            #i += 1
        print(f"loss on epoch {epoch} = {total_loss}")
        val_score = Val(val_loader, model)
        print(f"val_score on epoch {epoch} = {val_score}")
        if val_score <= last_val_score: break
        last_val_score = val_score
    return model




if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Need model description")
    else:
        X_train, X_test, X_val, Y_train, Y_test, Y_val, weight_pos = utils.load_data_nn(DATA_FILE, NUM_TASKS)

        # train_data = d.TensorDataset(torch.LongTensor(X_train), torch.FloatTensor(Y_train))
        # test_data = d.TensorDataset(torch.LongTensor(X_test), torch.FloatTensor(Y_test))
        # val_data = d.TensorDataset(torch.LongTensor(X_val), torch.FloatTensor(Y_val))

        train_data = EmojiDataset(X_train, Y_train)
        test_data = EmojiDataset(X_test, Y_test)
        val_data = EmojiDataset(X_val, Y_val)

        train_loader = d.DataLoader(train_data, batch_size=BATCH_SIZE)
        test_loader = d.DataLoader(test_data, batch_size=BATCH_SIZE)
        val_loader = d.DataLoader(val_data, batch_size=BATCH_SIZE)
        weight_pos = torch.tensor(weight_pos)
        model = Train(train_loader, val_loader, weight_pos)
        Test(test_loader, model)
