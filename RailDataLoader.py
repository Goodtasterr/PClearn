import os
from torch.utils.data import Dataset
import numpy as np
import torch

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class RailNormalDataset(Dataset):
    def __init__(self,root='/home/hwq/dataset/labeled/datanpy',npoints=25000,split='train'):
        self.root = root
        self.npoints = npoints

        fns = sorted(os.listdir(root))
        train_index = np.load('/home/hwq/dataset/labeled/trainindex.npy')
        test_index = np.load('/home/hwq/dataset/labeled/testindex.npy')

        alldatapath=[]
        for fn in fns:
            alldatapath.append(os.path.join(root,fn))

        self.datapath=[]
        if split == 'train':
            for index in train_index:
                self.datapath.append(alldatapath[index])
        if split == 'test':
            for index in test_index:
                self.datapath.append(alldatapath[index])


    def __getitem__(self, item):
        fn = self.datapath[item]
        data = np.load(fn).astype(np.float32)
        point_set = data[:,:3]
        seg = data[:, -1].astype(np.int32)

        point_set = pc_normalize(point_set)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice]
        seg = seg[choice]
        cls = np.zeros([1,]).astype(np.int32)
        return point_set,cls, seg

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    bs =5
    train_dataset = RailNormalDataset(split='train')
    test_dataset = RailNormalDataset(split='test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=bs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=bs)

    for point_set,cls, seg_num in test_dataloader:
        print((point_set).shape,cls.shape,seg_num.shape,seg_num.device)
        exit()
