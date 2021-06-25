
"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

import config
torch.set_printoptions(2)
""" 
Information about architecture config:
Tuple is structured by (out_channels, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4], 
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
  def __init__(self,in_channels,out_channels,bn_act = True,**kwargs):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.bn_act = bn_act
    self.cnn = nn.Conv2d(self.in_channels,self.out_channels,bias = not bn_act, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels)
    self.leaky = nn.LeakyReLU(0.1)

  def forward(self,X):
    if self.bn_act:
      return self.leaky(self.bn(self.cnn(X)))
    else:
      return self.cnn(X)

class ResidualBlock(nn.Module):
  def __init__(self,in_channels, use_residual= True,num_repeat  = 1):
    super().__init__()
    self.in_channels= in_channels
    self.use_residual= use_residual
    self.num_repeat = num_repeat
    self.module= nn.ModuleList([nn.Sequential(
      nn.Conv2d(in_channels,in_channels//2,kernel_size=1 ),
      nn.Conv2d(in_channels//2, in_channels,kernel_size=3, padding = 1)
    ) for _ in range(self.num_repeat)])


  def forward(self,X):
    for layer in self.module:
      if self.use_residual:
        X= X+layer(X)
      else:
        X = layer(X)
    return X

class ScalePrediction(nn.Module):
  def __init__(self, in_channels,num_classes):
    super().__init__()
    self.module = nn.Sequential(
      nn.Conv2d(in_channels, in_channels*2, kernel_size = 3, padding =1),
      nn.Conv2d(in_channels*2, (num_classes+5)*3,kernel_size = 1)
    )
    self.num_classes = num_classes
  def forward(self,X):
    return self.module(X).reshape(X.shape[0],3,self.num_classes+5, X.shape[2],X.shape[3]).permute(0,1,3,4,2)

class YOLOv3(nn.Module):
  def __init__(self, in_channels = 3, num_classes=20):
    super().__init__()
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.model = self.createModel()
    
  def forward(self,X):
    outputs,cat_input=[],[]

    for i,layer in enumerate(self.model):

      if isinstance(layer,ScalePrediction):
        outputs.append(layer(X))
        continue
      X = layer(X)
      
      if isinstance(layer,ResidualBlock):

        if layer.num_repeat == 8:
          cat_input.append(X)
      elif isinstance(layer,nn.Upsample):
        X = torch.cat([X,cat_input[-1]], dim = 1)
        cat_input.pop()
      # print(i,X.shape,layer)
    return outputs
  def createModel(self):
    models = nn.ModuleList()
    in_channels = self.in_channels
    for module in config:
      # print(model)
      if isinstance(module,tuple):

        out_channels,kernel_size,stride = module
        models.append(CNNBlock(in_channels, out_channels,kernel_size= kernel_size,
                          stride = stride, padding = 1 if kernel_size==3 else 0))
        in_channels = module[0]

      elif isinstance(module,list):
        models.append(ResidualBlock(in_channels,num_repeat = module[1]))
      elif isinstance(module,str):
        if module == "S":
          models+=[
            ResidualBlock(in_channels,use_residual =False),
            CNNBlock(in_channels,in_channels//2,kernel_size=1),
            ScalePrediction(in_channels//2,self.num_classes )
          ]
          in_channels= in_channels//2
        elif module =="U":
          models.append(nn.Upsample(scale_factor=2))
          in_channels = in_channels*3
    return models
if __name__ == "__main__":
  model = YOLOv3()
  model = model.to(torch.device("cuda:0"))
  X = torch.randn((2,3,416,416))
  X = X.to(torch.device("cuda:0"))
  Y= model(X)
  for y in Y:
    print(y*100)
    break


