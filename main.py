import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = torch.device(dev)
print(dev)

vec = [];
with open(r"C:\Users\jw\Downloads\nasdaq.csv") as csv_file:
    cr = csv.reader(csv_file, delimiter=',')
    n_lines = 0
    for cur in cr:
        vec.append(cur[4])

vec = vec[1:]
for i in range(0, len(vec)):
    vec[i] = float(vec[i])

n_test = 30
train_data = vec[:-n_test]
test_data = vec[-n_test:]

print("train size: " + str(len(train_data)))
print("test size: " + str(len(test_data)))
print(test_data)

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(np.array(train_data).reshape(-1, 1))

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_data_normalized = train_data_normalized.to(dev);
print("used device: " + str(train_data_normalized.device));
train_window = 30

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
print("train seq size: " + str(len(train_inout_seq)))
print(train_inout_seq[:5])

model = LSTM().to(dev);
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

fut_pred = 30

test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)

epochs = 150
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(dev),
                             torch.zeros(1, 1, model.hidden_layer_size).to(dev))

        y_pred = model(seq).to(dev)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if 1 + 1 == 2:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    seq = seq.to(dev)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(dev),
                        torch.zeros(1, 1, model.hidden_layer_size).to(dev))
        test_inputs.append(model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
print(actual_predictions)
