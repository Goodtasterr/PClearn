import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import copy
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

def pcd_test2():
    root = '/media/hwq/g/datasets/PointCloudTest/'
    output_root = '/media/hwq/g/datasets/pcl_test/'

    files = os.listdir(root)
    files.sort()
    print((files))

    # 1.增加一维　　２．保存为.npy　　　　　　　　　　　　　　　　　　
    for file in files[:-2]:
        file_name = file.split('_')[-1]
        file_rename = file_name.replace('.', 'v', 1).split('.')[0]
        print(root+file)
        #read .pcd
        pcd = o3d.io.read_point_cloud(root+file,format='pcd')

        points_np = np.asarray(pcd.points)
        points_number = points_np.shape[0]
        points_np_label = np.concatenate([points_np,np.zeros((points_number,1))],axis=-1)
        # print(points_np_label[:10])
        # o3d.visualization.draw_geometries([pcd], window_name='1',
        #                                   width=900, height=800)


        np.save(os.path.join(output_root, file_rename), points_np_label)
        # exit()
        # print(file_rename)


def pc_colors(arr):
    list = np.asarray([
        # [255,0,0],
        # [0,255,0],
        # [0,0,255],
        # [127, 0, 127],
        # [127, 127, 0]
        [255, 0, 0],  # 0 unlabeled
        [0, 0, 255],  # 1 car
        [244, 250, 88],  # 2 bicycle
        [138, 41, 8],  # 3 motorcycle
        [180, 4, 49],  # 4 truck
        [255, 0, 0],  # 5 other-vehicle
        [0, 0, 255],  # 6 person
        [191, 0, 255],  # 7 bicyclist
        [138, 41, 8],  # 8 motorcyclist
        [254, 46, 247],  # 9 road
        [245, 169, 242],  # 10 parking
        [219, 169, 1],  # 11 side walk
        [75, 8, 138],  # 12 other-ground
        [46, 204, 250],  # 13 building
        [0, 128, 255],  # 14 fence
        [49, 180, 4],  # 15 vegetation
        [8, 41, 138],  # 16 trunk
        [46, 254, 154],  # 17 terrain
        [169, 245, 242],  # 18 pole
        [0, 0, 255],  # 19 traffic-sign
    ])
    colors = []
    for i in arr:
        colors.append(list[i])

    return np.asarray(colors)/255

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
    root = '/media/hwq/g/dataset/label2'
    save_dir = '/media/hwq/g/dataset'

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
    np.save(save_dir+'/trainindex',train_list_np)

    np.save(save_dir+'/testindex',test_list_np)

    train_npy = np.load(save_dir+'/trainindex.npy')
    print('train list:',train_npy, len(train_npy))
def pc_label(points,number,number2,ranges):
    '''
    :param points: nparray:[N,3]
    :param number: nparray:[n]:classes index
    :param range: list:[n,6] n:classees; 6:x_min,x_max,y_min,y_max,z_min,z_max
    :return: [N',3]
    '''
    # global bbb
    label = np.zeros([points.shape[0]]).astype(int)
    part_index = []
    for i,range in enumerate(ranges):
        ranged = copy.deepcopy(range)
        if i ==0:
            ranged[3] = range[3]+number2[0]*(points[:,2]**number2[2])
            ranged[2] = ranged[3]-2.2
        else:
            ranged[3] = range[3] - number2[0] * (points[:, 2])
            ranged[2] = range[2] - number2[0] * (points[:, 2])
        index_x = (points[:, 0] > (ranged[0])) & (points[:, 0] < ranged[1])
        index_y = (points[:, 1] > (ranged[2])) & (points[:, 1] < ranged[3])
        index_z = (points[:, 2] > (ranged[4])) & (points[:, 2] < ranged[5])
        index = index_x & index_y & index_z
        part_index.append(index)
        index2 = np.asarray(index).astype(int)
        index2 = index2*number[i]
        label+=(index2)
    return label, part_index

if __name__ == '__main__':
    root = '/home/hwq/dataset/raildatarename'
    save_dir = '/home/hwq/dataset/labeled'
    data_root = '/home/hwq/dataset/labeled/data'
    label_root = '/home/hwq/dataset/labeled/label2'
    parameter_root = '/home/hwq/dataset/labeled/parameter2'
    data_txt = '/home/hwq/dataset/labeled/datanpy'


    files = os.listdir(root)
    files.sort()
    print(len(files))
    range = [-5.5, -2.4, -1, 1, 6.5, 50] #下 上 左 右 前 后 max[-5.5, 7, -30, 30, 6.5, 70]

    #多目标 位置区间
    end =43
    endend = 47
    ranges = [[-5.5, -2.4, -1.3,.8, 6.5, 70],  #轨道
              # [0.0, 6.5, 4.2, 5.3, 6, 11],  # 电线杆右2
              [-.3,6.5, 1.5, 8, 6, endend],  # 电线杆右1

              [-.3, 7, -3.6, 1.5, 6.5, endend],  # 电线杆中
              [-.8, 7, -6.5, -3.6, 6, 70],  # 电线杆左

              [-.5, 7, -30, -6.5, 6.5, 50],  # 电线杆左中
              [-.9, 7, -34, -6.5, 50, 70],  # 电线杆左前
              ]
    number = [1,2,2,2,2,2]
    # k,_,直/弯道,_,_  直道是一次函数，弯道是二次函数
    number2 = [-2/70,0,1.5,0,0,0]
    ranges_np = np.asarray(ranges)
    number_np = np.asarray(number)
    number_np = number_np[:,np.newaxis]
    number2_np = np.asarray(number2)
    number2_np = number2_np[:,np.newaxis]
    print('num',number_np.shape,number2_np.shape)
    parameter = np.concatenate((ranges_np,number_np),axis=-1)
    parameter = np.concatenate((parameter,number2_np),axis=-1)
    for i,file in enumerate(files):
        #1370 开始 弯道
        if i >1560:
            print('i =',i)
            file_name = file.split('.')[0]
            pcd = o3d.io.read_point_cloud(os.path.join(root,file))
            points = np.asarray(pcd.points)
            label,part_index = pc_label(points,number,number2,ranges)
            part_points = []
            show_all = []

            for part_i in part_index:
                part_point = points[part_i]
                # part_points.append(part_point)
                pcd_part = o3d.geometry.PointCloud()
                pcd_part.points = o3d.utility.Vector3dVector(part_point[:, 0:3])
                aabb = pcd_part.get_axis_aligned_bounding_box()
                aabb.color = (0, 0, 0)
                show_all.append(aabb)

            # print(type(label))
            new_data = np.concatenate((points,label[:, np.newaxis]),axis=-1)

            colors = pc_colors(label)
            print(new_data.shape)
            pcd_new = o3d.geometry.PointCloud()
            pcd_new.points = o3d.utility.Vector3dVector(points[:, 0:3])
            pcd_new.colors = o3d.utility.Vector3dVector(colors.squeeze())
            show_all.append(pcd_new)
            # print(len(show_all))
            # exit()
            o3d.visualization.draw_geometries(show_all, window_name=file + '--' + str(i),
                                              width=900, height=800)
            # o3d.visualization.draw_geometries([pcd_new], window_name=file+'--'+str(i),
            #                                   width=900, height=800)
            np.save(os.path.join(label_root,file_name),new_data)
            np.savetxt(os.path.join(parameter_root,file_name)+'.txt',parameter,fmt='%0.2f')
            # exit()