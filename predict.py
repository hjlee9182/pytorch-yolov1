#encoding:utf-8
#
#created by xiongzihua
#
import torch
from torch.autograd import Variable
import torch.nn as nn

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np
from glob import glob
import sys

VOC_CLASSES =[     # always index 0
'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic_light',
    'fire_hydrant',
    'stop_sign',
    'parking_meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports_ball',
    'kite',
    'baseball_bat',
    'baseball_glove',
    'skateboard',
    'surfboard',
    'tennis_racket',
    'bottle',
    'wine_glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana'
    ,'apple'
    ,'sandwich'
    ,'orange'
    ,'broccoli'
,'carrot'
,'hot_dog'
,'pizza'
,'donut'
,'cake'
,'chair'
,'couch'
,'potted_plant'
,'bed'
,'dining_table'
,'toilet'
,'tv'
,'laptop'
,'mouse'
,'remote'
,'keyboard'
,'cell_phone'
,'microwave'
,'oven'
,'toaster'
,'sink'
,'refrigerator'
,'book'
,'clock'
,'vase'
,'scissors'
,'teddy_bear'
,'hair_drier'
,'toothbrush']


def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob*max_prob)
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.stack(cls_indexs,0) #(n,)
    keep = nms2(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]

def batch_iou(boxes,box):
    lr = np.maximum(
            np.minimum(boxes[:,0]+0.5*boxes[:,2],box[0]+0.5*box[2])-np.maximum(boxes[:,0]-0.5*boxes[:,2],box[0]-0.5*box[2]),0)
    tb = np.maximum(
            np.minimum(boxes[:,1]+0.5*boxes[:,3],box[1]+0.5*box[3])-np.maximum(boxes[:,1]-0.5*boxes[:,3],box[1]-0.5*box[3]),0)
    inter= lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] -inter
    return inter/union

def nms2(boxes,scores,threshold=0.5):
    order = torch.from_numpy(scores.argsort().numpy()[::-1].copy())
    keep = [True]*len(order)

    for i in range(len(order)-1):
        ovps = batch_iou(boxes[order[i+1:]],boxes[order[i]])
        for j,ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j+i+1]]=False
    return keep

def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
#
#start predict one image
#
def predict_gpu(model,image_name,root_path=''):

    result = []
    image = cv2.imread(root_path+image_name)
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mean = (123,117,104)#RGB
    img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    img = Variable(img[None,:,:,:],volatile=True)
    img = img.cuda()

    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    boxes,cls_indexs,probs =  decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])
    return result
        



if __name__ == '__main__':
    model = resnet50()
    print('load model...')
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    model.cuda()
    val_list = glob('../../data/images/val2017/*.jpg')
    num = 0
    for image_name in val_list:
        image = cv2.imread(image_name)
        txt = image_name.split('/')[-1][:-4]+'.txt'
        result = predict_gpu(model,image_name)
        for left_up,right_bottom,class_name,_,prob in result:
            if prob<0.5:
                continue
            label = class_name+str(round(prob,2))
            f = open('../mAP/input/detection-results/'+txt,'a')
            f.write(f'{class_name} {str(prob)} {str(left_up[0])} {str(left_up[1])} {str(right_bottom[0])} {str(right_bottom[1])}\n')
            f.close()
        print(num)
        num+=1




