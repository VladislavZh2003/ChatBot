import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import bag_of_words, tokenize, stem
from torchMode import NeuralNet

with open('trainingData.json', 'r') as f:
    purposes = json.load(f)

all_words = []
labels = []
xy = []
for purpose in purposes['purposes']:
    label = purpose['label']
    labels.append(label)
    for template in purpose['templates']:
        w = tokenize(template)
        all_words.extend(w)
        xy.append((w, label))

ignore_words = ['?', '.', '!',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
labels = sorted(set(labels))


print(len(xy), "templates")
print(len(labels), "labels:", labels)
print(len(all_words), "unique stemmed words:", all_words)

X_train = []
y_train = []
for (template_sentence, label) in xy:
    bag = bag_of_words(template_sentence, all_words)
    X_train.append(bag)
    labelB = labels.index(label)
    y_train.append(labelB)

X_train = np.array(X_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(labels)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labelsB) in train_loader:
        words = words.to(device)
        labelsB = labelsB.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labelsB)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"labels": labels
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')