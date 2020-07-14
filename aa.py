import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil

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
    index2 = np.asarray(index).astype(int)
    # print(index2)
    points_new = points[index]

    return points_new, index2

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

def file_produce():
    root = '/home/hwq/dataset/pcdfiltedall'
    save_dir = '/home/hwq/dataset/rangedpc'
    files = os.listdir(root)
    files.sort()
    print(len(files))
    pcl_name='test1'
    range = [-5.5, 7, -30, 30, 6.5, 70]

    for file in files:

        pcd = o3d.io.read_point_cloud(os.path.join(root,file))

        points = np.asarray(pcd.points)
        points_new,index = pc_reduce(points,range)


        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(points_new)
        o3d.io.write_point_cloud(os.path.join(save_dir,file), pcd_new)
        # exit()


        # o3d.visualization.draw_geometries([pcd],window_name='rawdata',
        #                               width=800,height=600)
        # o3d.visualization.draw_geometries([pcd_new],window_name='ranged',
        #                               width=800,height=600)

def prp_2_oh_array(arr):
    """
    概率矩阵转换为OH矩阵
    arr = np.array([[0.1, 0.5, 0.4], [0.2, 0.1, 0.6]])
    :param arr: 概率矩阵
    :return: OH矩阵
    """
    arr_size = arr.shape[1]  # 类别数
    arr_max = np.argmax(arr, axis=1)  # 最大值位置
    print(arr_max)
    print(np.eye(arr_size))
    oh_arr = np.eye(arr_size)[arr_max]  # OH矩阵
    return oh_arr

def labeled_data():
    root = '/home/hwq/dataset/raildatarename'
    save_dir = '/home/hwq/dataset/labeled'
    data_root = '/home/hwq/dataset/labeled/data'
    label_root = '/home/hwq/dataset/labeled/label'
    data_txt = '/home/hwq/dataset/labeled/datanpy'


    files = os.listdir(root)
    files.sort()
    print(len(files))
    range = [-5.5, -1, -1, 1, 6.5, 50]

    for file in files:
        file_name = file.split('.')[0]
        pcd = o3d.io.read_point_cloud(os.path.join(root,file))
        points = np.asarray(pcd.points)
        points_new,index = pc_reduce(points,range)
        new_data = np.concatenate((points,index[:, np.newaxis]),axis=-1)
        np.save(os.path.join(data_txt,file_name),new_data)
        exit()
def sort_train_test():
    root = '/home/hwq/dataset/raildatarename'
    save_dir = '/home/hwq/dataset/labeled'

    import random

    files = os.listdir(root)
    files.sort()
    print(len(files))
    index_list = list(range(len(files)))
    train_list = random.sample(index_list,int(0.75*len(files)))
    train_list.sort()
    print(len(train_list),(train_list))
    test_list = list(set(index_list).difference(set(train_list)))
    test_list.sort()
    print(len(test_list),(test_list))

    train_list_np = np.asarray(train_list).astype(np.int)
    test_list_np = np.asarray(test_list).astype(np.int)
    # np.save(save_dir+'/trainindex',train_list_np)

    # np.save(save_dir+'/testindex',test_list_np)

    train_npy = np.load(save_dir+'/trainindex.npy')
    print('train list:',train_npy, len(train_npy))

if __name__ == '__main__':
    root = '/home/hwq/dataset/raildatarename'
    save_dir = '/home/hwq/dataset/labeled'
    data_root = '/home/hwq/dataset/labeled/data'
    label_root = '/home/hwq/dataset/labeled/label'
    data_txt = '/home/hwq/dataset/labeled/datanpy'

    import random



    # print('list',index_list)
    files = os.listdir(root)
    files.sort()
    fns = sorted(os.listdir(root))
    # fns.sort()
    print('fns',len(fns),fns,)

    # for file in files:
    #     file_name = file.split('.')[0]
    #     file_npy = file_name+'.npy'
    #     # file_txt = file_name+'.txt'
    #     # print(file_txt)
    #     pcd = o3d.io.read_point_cloud(os.path.join(root,file))
    #
    #     points = np.asarray(pcd.points)
    #
    #     points_new,index = pc_reduce(points,range)
    #     new_data = np.concatenate((points,index[:, np.newaxis]),axis=-1)
    #     # print(new_data[:10])
    #     # np.save(os.path.join(data_txt,file_name),new_data)
    #     data_npy = np.load(os.path.join(data_txt,file_npy))
    #     # print(data_npy[:10])
    #     # exit()
    #
    #
    #     # print('index',type(index),index.shape)
    #     # # print(new_data[16000:16020])
    #     # # exit()
    #     colors_list = data_npy[:,-1].astype(int)
    #     # num_points = points_new.shape[0]
    #     colors = pc_colors(colors_list)
    #     # pcd.colors = o3d.utility.Vector3dVector(colors)
    #     # print(colors.shape)
    #     pcd_new = o3d.geometry.PointCloud()
    #     pcd_new.points = o3d.utility.Vector3dVector(data_npy[:,:3])
    #
    #
    #
    #     pcd_new.colors = o3d.utility.Vector3dVector(colors)
    #     # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    #     # pcd.colors[1500] = [1, 0, 0]
    #     # o3d.io.write_point_cloud(os.path.join(save_dir,file), pcd_new)
    #     o3d.visualization.draw_geometries([pcd_new],window_name='ranged',
    #                                   width=800,height=600)
    #     exit()