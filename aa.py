import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def pc_reduce(points,range):
    '''

    :param points: nparray:[N,3]
    :param range: list:[6] x_min,x_max,y_min,y_max,z_min,z_max
    :return: [N',3]
    '''
    index_x = (points[:, 0] > (range[0])) & (points[:, 0] < range[1])
    index_y = (points[:, 1] > (range[2])) & (points[:, 1] < range[3])
    index_z = (points[:, 2] > (range[4])) & (points[:, 2] < range[5])
    index = index_x & index_y & index_z
    points_new = points[index]

    return points_new

if __name__ == '__main__':
    root = '/home/hwq/dataset/pcdfiltedall'
    files = os.listdir(root)
    print(len(files))
    pcl_name='test1'
    pcd = o3d.io.read_point_cloud(os.path.join(root,files[745]))

    range = [-5.5,7,-40,40,6.5,70]

    points = np.asarray(pcd.points)
    points_new = pc_reduce(points,range)
    print(points_new.shape)

    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points_new)
    o3d.visualization.draw_geometries([pcd],window_name='rawdata',
                                      width=800,height=600)
    o3d.visualization.draw_geometries([pcd_new],window_name='ranged',
                                      width=800,height=600)

    max = np.max(points,0)
    min = np.min(points,0)
    print(points.shape,max,min)