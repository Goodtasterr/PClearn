import os
from torch.utils.data import Dataset
import numpy as np
import torch
from os.path import join

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
        train_index = np.load(os.path.join(root,'..')+'/trainindex.npy')
        test_index = np.load(os.path.join(root,'..')+'/testindex.npy')

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
        seg = data[:, -1].astype(np.int64)

        point_set = pc_normalize(point_set)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice]
        seg = seg[choice]+1
        # cls = np.zeros([1,]).astype(np.int32)
        return point_set, seg

    def __len__(self):
        return len(self.datapath)

class raildata_RandLA(Dataset):
    def __int__(self,mode):
        self.name = 'raildata_RandLA'
        self.dataset_path = '/home/hwq/dataset/labeled/datanpy'
        self.label_to_names = {0:'unlabeled',
                               1:'rail',
                               2:'pole'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()]) # [0,1,2]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)} # dict {0:0,1:1,2:2}
        self.ignored_labels = np.sort([0])

        fns = sorted(os.listdir(root))
        train_index = np.load(os.path.join(root,'..')+'/trainindex.npy')
        test_index = np.load(os.path.join(root,'..')+'/testindex.npy')

        alldatapath=[]
        for fn in fns:
            alldatapath.append(os.path.join(root,fn))

        self.data_list=[]
        if mode == 'train':
            for index in train_index:
                self.data_list.append(alldatapath[index])
        if mode == 'test':
            for index in test_index:
                self.data_list.append(alldatapath[index])

    def get_data(self, file_path):
        seq_id = file_path.split('/')[-3]
        frame_id = file_path.split('/')[-1][:-4]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        # Read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # Load labels
        if int(seq_id) >= 11: #11-21 is testing set
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)
        else:
            label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
            labels = np.squeeze(np.load(label_path))
        return points, search_tree, labels

if __name__ == '__main__':
    bs =5
    root = '/home/hwq/dataset/labeled'
    print(root+'/datanpy')
    train_dataset = RailNormalDataset(split='train')
    test_dataset = RailNormalDataset(split='test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=bs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=bs)

    for point_set, seg_num in test_dataloader:
        print((point_set)[0,:10],(seg_num.min()),seg_num.device)
        exit()
