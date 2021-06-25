# import torch
# from torch import nn as nn
# from utils import IoU
# class YOLOloss(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.entropy = nn.CrossEntropyLoss()
#     self.bce = nn.BCEWithLogitsLoss()
#     self.mse = nn.MSELoss()
#     self.sigmoid = nn.Sigmoid()
    
#     self.lambda_class = 1
#     self.lambda_noobj = 10
#     self.lambda_obj = 1
#     self.lambda_box = 10
  
#   def forward(self,predictions,target,anchors):
#     obj = target[...,0]==1
#     noobj = target[...,0]==0

#     # noobj loss function
#     no_obj_loss =self.bce(predictions[...,0:1][noobj],target[...,0:1][noobj])

#     #obj loss function
#     anchors = anchors.reshape(1,3,1,1,2)
#     obj_pred = torch.cat([self.sigmoid(predictions[...,1:3]),torch.exp(predictions[...,3:5])*anchors],dim = -1)
#     ious = IoU(obj_pred[obj],target[...,1:5][obj]).detach()
#     obj_loss = self.mse(self.sigmoid(predictions[...,0:1][obj]),ious*target[...,0:1][obj])

#     # box loss function
#     predictions[...,1:3] = self.sigmoid(predictions[...,1:3])
#     target[...,3:5] = torch.log(1e-6+target[...,3:5]/anchors)
#     box_loss = self.mse(predictions[...,1:5][obj],target[...,1:5][obj])

#     # class loss function
#     class_loss = self.entropy(predictions[...,5:][obj],target[...,5][obj].long())

#     return (no_obj_loss*self.lambda_noobj+
#             obj_loss*self.lambda_obj+
#             box_loss * self.lambda_box +
#             class_loss * self.lambda_class)
import random
import torch
import torch.nn as nn

from utils import IoU,intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
