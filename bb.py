import numpy as np #1
import cv2 #2
import os





def show_box(img_root,label_root):

    canvas = cv2.imread(img_root,1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = np.loadtxt(label_root).reshape(-1, 9)

    nodes = []
    cls = []
    for i in range(len(label)):
        cls.append(label[i][0])
        for j in range(4):
            nodes.append(label[i][1 + 2 * j:3 + 2 * j].astype(np.int))
    print(cls)

    red = (0, 0, 255)  # 8
    colors = [(0, 0, 255), #red
              (0, 255, 0), #green
              (255, 0, 0), #blue
              (0, 0, 255)]
    for i in range(len(nodes)):
        if i > 0 and (i + 1) % 4 == 0:
            cv2.line(canvas, tuple((nodes[i])), tuple((nodes[i - 3])), colors[i%4], 3)  # 9node
            canvas = cv2.putText(canvas, str(cls[int((i+1)/4)-1]), tuple((nodes[i])), font, 1.2, (255, 255, 255), 2)
        else:
            cv2.line(canvas, tuple((nodes[i])), tuple((nodes[i + 1])), colors[i%4], 3)  # 9node

    img_test2 = cv2.resize(canvas, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('resize1', img_test2)
    cv2.waitKey()
    # cv2.imshow("Canvas", canvas)  # 10
    # cv2.waitKey(0)  # 11
    cv2.destroyAllWindows()


if __name__ == '__main__':

    img_root = '/media/hwq/g/datasets/rssj4/images/'
    label_root = '/media/hwq/g/datasets/rssj4/labels'

    img_files = sorted(os.listdir(img_root))
    img_files.sort(key=lambda x: int(x[:-4]))
    label_files = (os.listdir(label_root))
    label_files.sort(key=lambda x: int(x[:-4]))
    cls = []
    for i in range(len(img_files)):
        print(os.path.join(img_root,img_files[i]))
        print(os.path.join(label_root,label_files[i]))
        # exit()

    #     label = np.loadtxt(os.path.join(label_root,label_files[i])).reshape(-1, 9)
    #
    #
    #     for i in range(len(label)):
    #         cls.append(label[i][0])
    # label_np = np.asarray(cls)
    # print(label_np.shape)
    # print(len(label_np),max(label_np),min(label_np))
    # for i in range(6):
    #     print(sum((label_np==i)))

        show_box(os.path.join(img_root,img_files[i]),
                 os.path.join(label_root,label_files[i]))