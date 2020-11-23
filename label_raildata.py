import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import copy

def pc_colors(arr):
    list = np.asarray([
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [127, 0, 127],
        [127, 127, 0]
    ])
    colors = []
    for i in arr:
        colors.append(list[i])

    return np.asarray(colors)/255

def label_range(points,number,number2,ranges):
    '''
    :param points: nparray:[N,3]
    :param number: nparray:[n]:classes number
    :param number2: nparray:[n]:k,_,n,_,_..., shape is same as number
    :param range: list:[n,6] n:classees; 6:x_min,x_max,y_min,y_max,z_min,z_max
    :return: [N',3]
    '''
    # global bbb
    label = np.zeros([points.shape[0]]).astype(int)
    part_index = []
    for i,range in enumerate(ranges):
        ranged = copy.deepcopy(range)
        if i ==0: 
            ranged[3] = range[3]+number2[0]*(
                    points[:,2]**2) + number2[2]*points[:,2]
            ranged[2] = ranged[3] - 1.5/(((number2[0]*points[:,2]+number2[2])**2+1)**(1/2))

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

def label_rail_data():
    #point cloud file root
    root = '/home/hwq/dataset/raildatarename'
    #labeled point file root
    label_root = '/home/hwq/dataset/labeled/label2'
    #label parameter save root
    parameter_root = '/home/hwq/dataset/labeled/parameter2'

    files = sorted(os.listdir(root))
    range = [-5.5, -2.4, -1, 1, 6.5, 50] #下 上 左 右 前 后 max[-5.5, 7, -30, 30, 6.5, 70]

    #多目标 位置区间
    ranges = [[-5.5, -2.4, -1.7,0.4, 6.5, 70],  #轨道
              # [0.0, 6.5, 4.2, 5.3, 6, 11],  # 电线杆右2
              [-1.7,6.5, -3, 6, 6,45],  # 电线杆右1

              [1.4, 7, -17, -3, 6.5, 65],  # 电线杆中
              [-1.8, 7, -22, -17, 26, 70],  # 电线杆左

              [-1.8, 7, -23, -17, 53, 53],  # 电线杆左中
              [-1.8,6.5, -3, 9, 45,70],  # 电线杆左前
              ]
    number = [1,2,2,2,2,2] #label parameter 1 : label value

    # [a,_,b,_,_]  直道是一次函数n=1，弯道是二次函数n>1
    #shape of number is same as number2 and range.shape[1]
    number2 = [0.05/70,0,4.5/70,0,0,0] #label parameter 2 : y=a*z^2 + b*z

    for i,file in enumerate(files):
        #1370 开始 弯道   1929-2252 复杂
        if i >=1930: #当前帧数
            print('id =',i)
            file_name = file.split('.')[0]
            pcd = o3d.io.read_point_cloud(os.path.join(root,file))
            points = np.asarray(pcd.points)
            label,part_index = label_range(points,number,number2,ranges)
            part_points = []
            show_all = []

            for part_i in part_index:
                print('test1', points.shape,part_i.shape)
                part_point = points[part_i]
                pcd_part = o3d.geometry.PointCloud()
                pcd_part.points = o3d.utility.Vector3dVector(part_point[:, 0:3])
                aabb = pcd_part.get_axis_aligned_bounding_box()
                aabb.color = (0, 0, 0)
                show_all.append(aabb)

            new_data = np.concatenate((points,label[:, np.newaxis]),axis=-1)

            colors = pc_colors(label)
            print(new_data.shape)
            pcd_new = o3d.geometry.PointCloud()
            pcd_new.points = o3d.utility.Vector3dVector(points[:, 0:3])
            pcd_new.colors = o3d.utility.Vector3dVector(colors.squeeze())
            show_all.append(pcd_new)

            o3d.visualization.draw_geometries(show_all, window_name=file + '--' + str(i),
                                              width=1080, height=1920)
            #save label parameter
            np.save(os.path.join(label_root,file_name),new_data)

            ranges_np = np.asarray(ranges)
            number_np = np.asarray(number)
            number_np = number_np[:, np.newaxis]
            number2_np = np.asarray(number2)
            number2_np = number2_np[:, np.newaxis]
            parameter = np.concatenate((ranges_np, number_np), axis=-1)
            parameter = np.concatenate((parameter, number2_np), axis=-1)
            np.savetxt(os.path.join(parameter_root,file_name)+'.txt',parameter,fmt='%0.2f')
            # exit()

if __name__ == '__main__':
    label_rail_data()