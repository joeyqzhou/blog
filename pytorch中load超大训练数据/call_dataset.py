import torch
from Dataset import Dataset

arg = "HOO"
train_set = Dataset("train.txt", arg)
params = {'batch_size': 2,
              'shuffle': True,
              'pin_memory': True,
              'num_workers': 8,
              'drop_last' : True}

training_generator = torch.utils.data.DataLoader(train_set, **params)

for x in training_generator:
    print(x)
    print("-----")
