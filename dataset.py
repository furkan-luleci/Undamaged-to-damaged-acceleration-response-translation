import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from natsort import natsorted

class Dataset(Dataset):
    def __init__(self, root_a0, root_a1): #a0(undamaged) a1(damaged)
        self.root_a0 = root_a0
        self.root_a1 = root_a1

        self.a0_data = natsorted(os.listdir(root_a0))
        self.a1_data = natsorted(os.listdir(root_a1))

        self.length_dataset = max(len(self.a0_data), len(self.a1_data)) 
        self.a0_len = len(self.a0_data)
        self.a1_len = len(self.a1_data)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        a0 = self.a0_data[index % self.a0_len]
        a1 = self.a1_data[index % self.a1_len]

        a0_path = os.path.join(self.root_a0, a0)
        a1_path = os.path.join(self.root_a1, a1)
     
        a0 = np.array(pd.read_csv(a0_path, header=None))
        a1 = np.array(pd.read_csv(a1_path, header=None))


        return a0,a1