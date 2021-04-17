import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

""" Necessary args """
filename = r"C:\Users\jw\Downloads\appl.csv"
outfile_name = r"C:\Users\jw\Downloads\appl_pred.csv"
train_data_ratio = 0.1
epochs = 10000
train_window = 100

""" Optional args """
size_limit = None
begin_after_this = None

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=30, output_size=1):
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

vec = []
with open(filename) as csv_file:
    cr = csv.reader(csv_file, delimiter=',')
    i = 0
    for cur in cr:
        i += 1
        if i == 1:
            continue
        vec.append(float(cur[1][1:]))

vec.reverse()
orig_vec = vec

if begin_after_this:
    if len(vec) >= begin_after_this:
        vec = vec[begin_after_this:]

if size_limit:
    if len(vec) > size_limit:
        vec = vec[:size_limit]

test_idx = int(len(vec) * train_data_ratio)
train_data = vec[:-test_idx]
test_data = vec[-test_idx:]

print("train size: " + str(len(train_data)))
print("test size: " + str(len(test_data)))

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(np.array(train_data).reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_data_normalized = train_data_normalized.to(dev)
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

test_data_normalized = scaler.fit_transform(np.array(test_data).reshape(-1, 1))
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)
test_data_normalized = test_data_normalized.to(dev)
test_inputs = train_data_normalized[-train_window:].tolist()

print("train seq size: " + str(len(train_inout_seq)))
print("test input size: " + str(len(test_inputs)))

model = LSTM().to(dev)
loss_function = nn.MSELoss().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("@@@@@@@@@ printing model @@@@@@@@@@")
print(model)
print("@@@@@@@@@ printed model @@@@@@@@@@")

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(dev),
                             torch.zeros(1, 1, model.hidden_layer_size).to(dev))

        seq = seq.to(dev)
        labels = labels.to(dev)

        y_pred = model(seq).to(dev)

        single_loss = loss_function(y_pred, labels).to(dev)
        single_loss.backward()
        optimizer.step()

    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

model.eval()

for i in range(train_window):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    seq = seq.to(dev)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(dev),
                        torch.zeros(1, 1, model.hidden_layer_size).to(dev))
        test_inputs.append(model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

for e in actual_predictions:
        print(str(float(e)))

with open(outfile_name, 'w') as csv_file:
    cr = csv.writer(csv_file, delimiter=' ')
    for e in orig_vec:
        cr.writerow(str(float(e)))
    cr.writerow('----------------')
    for e in actual_predictions:
        cr.writerow(str(float(e)))