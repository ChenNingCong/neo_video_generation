import torch
class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_l):
        self.dataset = dataset
        self.base_l = base_l
        self.l = len(self.dataset)
    def __len__(self):
        return self.l
    def __getitem__(self, i):
        return self.dataset[i %self.base_l]
