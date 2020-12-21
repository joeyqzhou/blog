import torch
from multi_process_argument import multi_process_line

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file, prefix): 
        f = open(file)
        self.data = multi_process_line(f, prefix) 
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index] 

    def __len__(self):
        return self.len
