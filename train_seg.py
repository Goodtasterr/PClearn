'''
Author: hwq
Date: Jul 2020
'''
import torch
from tqdm import tqdm
import os
import argparse
from RailDataLoader import RailNormalDataset
import logging
from pathlib import Path
import sys
import datetime
import importlib
from torch.utils.data import DataLoader
import shutil
import provider


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIT = BASE_DIR
sys.path.append(os.path.join(ROOT_DIT,'models'))

def to_categorical(y,num_classes):
    '''one-hot encode'''
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),].to(y.device)
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model',type=str,default='pointnet')
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--epoch',type=int,default=100)
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--log_dir',type=str,default='./logsave')
    parser.add_argument('--decay_rate',type=float,default=1e-4)
    parser.add_argument('--npoint',type=int,default=40000)
    parser.add_argument('--step_size',type=int,default=20)
    parser.add_argument('--lr_decay',type=float,default=0.5)

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt'%(log_dir,args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '/home/hwq/dataset/labeled/datanpy'

    Train_Dataset = RailNormalDataset(root=root,npoints=args.npoint,split='train')
    TrainDataLoader = DataLoader(Train_Dataset,batch_size=args.batch_size,
                                 shuffle=True,num_workers=4)
    Test_Dataset = RailNormalDataset(root=root,npoints=args.npoint,split='test')
    TestDataLoader = DataLoader(Test_Dataset,batch_size=args.batch_size,
                                shuffle=True,num_workers=4)
    log_string("The number of train data is: %d"%len(Train_Dataset))
    log_string("The number of test data is: %d"%len(Test_Dataset))

    num_classes = 1
    num_part = 2
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py'%args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py',str(experiment_dir))

    classifier = MODEL.get_model(num_part,normal_channel = False).cuda()
    criterion = MODEL.get_loss().cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    try:
        checkpoint = torch.load(str(experiment_dir)+'/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No esxisting model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9,0.999),
            eps=1e-08,
            weight_decay=args.decat_rate
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9
        )

    def bn_momentun_adjust(m,momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):'%(global_epoch+1, epoch+1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate*(args.lr_decay**(epoch//args.step_size)),LEARNING_RATE_CLIP)
        log_string('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL*(MOMENTUM_DECCAY**(epoch//MOMENTUM_DECCAY_STEP))
        if momentum<0.01:
            momentum=0.01
        print('BN momentum updated to: %f'%momentum)
        classifier = classifier.apply(lambda x: bn_momentun_adjust(x,momentum))

        '''learning one epoch'''
        for i, data in tqdm(enumerate(TrainDataLoader),total=len(TrainDataLoader),smoothing=0.9):
            points, label, target = data
            points = points.data.numpy()
            points[:,:,0:3] = provider.random_scale_point_cloud(points[:,:,0:3])
            points[:,:,0:3] = provider.shift_point_cloud(points[:,:,0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(),label.long().cuda(), target.long().cuda()
            points = points.transpose(2,1).contiguous()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points, to_categorical(label,num_classes))
