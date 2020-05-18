#%%
import pickle
import numpy as np 
from torch.utils.data import DataLoader,Dataset

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class MyDataset(DataLoader):
    def __init__(self):
        data = unpickle('../cifar-10-python/cifar-10-batches-py/data_batch_1')
        self.images = data[b'data']
        self.labels = data[b'labels']
        
        for i in range(2,6):
            data = unpickle('../cifar-10-python/cifar-10-batches-py/data_batch_'+str(i))
            nimages = data[b'data']
            nlabels = data[b'labels']
            self.images = np.concatenate((self.images, nimages),axis=0)
            self.labels.extend(nlabels)
                     
        self.length = len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].reshape(3,32,32)
        label = self.labels[index]
        return (image, label)

    def __len__(self):
        return self.length





# %%
