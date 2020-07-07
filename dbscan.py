import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os
if __name__ == '__main__':
    root = '/home/hwq/dataset/pcdfiltedall'
    files = os.listdir(root)
    print(len(files))

    eps=2
    pcl_name='test1'
    cmap = plt.get_cmap('Set2')

    pcd = o3d.io.read_point_cloud(os.path.join(root,files[1435]))
    print(pcd)
    t0 = time.time()
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=5))
    print(time.time()-t0)
    max_label = labels.max()
    print('%s has %d clusters' % (pcl_name, max_label + 1))

    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd],window_name=pcl_name,width=800,height=600)


