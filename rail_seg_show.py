import torch

from models.model import PointNetDenseCls, feature_transform_regularizer
import os
import numpy as np
import open3d   as o3d

'''PATH'''
weight_dir='./seg2/seg_model_Chair_24.pth'
data_dir = ''
root = '/media/hwq/g/datasets/label2'
test_index = np.load(os.path.join(root,'..')+'/testindex.npy').tolist()

fns = sorted(os.listdir(root))
print(len(fns))
'''MODEL'''
classifier = PointNetDenseCls(k=3)
weight = torch.load(weight_dir)
classifier.load_state_dict(weight)
classifier.cuda()
classifier=classifier.eval()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def pc_colors(arr):
    list = np.asarray([
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ])
    colors = []
    for i in arr:
        colors.append(list[i])

    return np.asarray(colors)

for i, file in enumerate(fns):
    if i in test_index:
        if i >100:
            data = np.load(os.path.join(root,file)).astype(np.float32)
            points = data[:,0:3]
            points = pc_normalize(points)
            points = points[np.newaxis,:,:]
            points = torch.from_numpy(points).transpose(2,1).contiguous().cuda()
            with torch.no_grad():
                seg,_,_ = classifier(points)
            print(seg.cpu().squeeze().shape)
            label = torch.max(seg.cpu().squeeze(),-1)[1]
            label= label.numpy()

            colors = pc_colors(label)
            pcd_new = o3d.geometry.PointCloud()
            pcd_new.points = o3d.utility.Vector3dVector(data[:,0:3])
            pcd_new.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd_new],window_name=file,
                                          width=800,height=600)

    # exit()
