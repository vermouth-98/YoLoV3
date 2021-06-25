import numpy as np
import os
from numpy.core.fromnumeric import argsort
import pandas as pd
import torch
import torchvision
import config
from torch.utils.data import DataLoader,Dataset
import utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class YOLODataset(torch.utils.data.Dataset):
  def __init__(self,csv_file,anchors,S,num_classes,augmentations =None,dir="PASCAL_VOC"):
    super(YOLODataset,self).__init__()
    self.dir = dir
    self.annotations = pd.read_csv(csv_file)
    self.mode = torchvision.io.image.ImageReadMode.RGB
    self.num_classes = num_classes
    self.anchors = torch.tensor(anchors[0]+anchors[1]+anchors[2])
    self.scale = S
    self.num_anchors = self.anchors.shape[0]
    self.anchors_per_scale = self.num_anchors//3
    self.augmentations =  augmentations
    self.ignore_iou_threshold = 0.5
  def __getitem__(self,idx):
    img = np.array(Image.open(os.path.join(self.dir,"images",self.annotations.iloc[idx,0])).convert("RGB"),dtype= np.uint8)
    bboxes = np.loadtxt(os.path.join(self.dir,"labels",self.annotations.iloc[idx,1]),delimiter=" ",ndmin = 2)
    bboxes = np.roll(bboxes,shift =4, axis =1).tolist()
    if self.augmentations:
      augmentations= self.augmentations(image = img,bboxes =bboxes)
      img = augmentations["image"]
      bboxes = augmentations["bboxes"]
    targets = self.createTargets(bboxes)
    return img,targets
  def __len__(self):
    return len(self.annotations)
  def createTargets(self,bboxes):
    
    target = [ torch.zeros((self.anchors_per_scale,S,S,6)) for S in self.scale]
    
    for bbox in bboxes:
      x,y,width,height,class_= bbox
  
      iou_wh = utils.Iou_wh(torch.tensor([bbox[:4]]),self.anchors).squeeze(0)
      max_idx = iou_wh.argsort(descending=True)

      has_scale = [False]*3
      for anchor_idx in max_idx:
        
        anchor_belong_scale = int(anchor_idx // self.anchors_per_scale)
        num_anchor_on_scale = int(anchor_idx % self.anchors_per_scale)
        S = self.scale[anchor_belong_scale]
        i,j = int(S*y),int(S*x)
        chosen_anchor = target[anchor_belong_scale][num_anchor_on_scale,i,j,0]
        if not chosen_anchor and not has_scale[anchor_belong_scale]:
          target[anchor_belong_scale][num_anchor_on_scale,i,j,0]=1 # object duoc gan
          has_scale[anchor_belong_scale]=True
          x_cell = S*x-j
          y_cell = S*y-i
          width_cell,height_cell = width*S, height*S
          box_coord = torch.tensor([x_cell,y_cell,width_cell,height_cell])
          target[anchor_belong_scale][num_anchor_on_scale,i,j,1:5] = box_coord
          target[anchor_belong_scale][num_anchor_on_scale,i,j,5]= int(class_)
        elif not chosen_anchor and iou_wh[anchor_idx]> self.ignore_iou_threshold:
          target[anchor_belong_scale][num_anchor_on_scale,i,j,0]=-1
      
    return target
def LoadData(csv_train,csv_val):
  train_data = YOLODataset(csv_train,config.ANCHORS,config.S,config.NUM_CLASSES,config.train_transforms)
  val_data  = YOLODataset(csv_train,config.ANCHORS,config.S,config.NUM_CLASSES,config.test_transforms)
  train_iter = torch.utils.data.DataLoader(
              train_data,
              batch_size=config.BATCH_SIZE,
              shuffle = True,
              pin_memory = config.PIN_MEMORY,
              drop_last = False,
              num_workers = 2)
  val_iter = torch.utils.data.DataLoader(
              val_data,
              batch_size=config.BATCH_SIZE,
              shuffle = True,
              pin_memory = config.PIN_MEMORY,
              drop_last =False,
              num_workers = 2)
  return train_iter,val_iter


if __name__ == '__main__':
  csv_file = "PASCAL_VOC/1examples.csv"
  data = YOLODataset(csv_file,config.ANCHORS,config.S,config.NUM_CLASSES,config.train_transforms)
  print(data[0])







