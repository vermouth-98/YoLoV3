import config
import torch
import collections
from tqdm import tqdm
def Iou_wh(boxes1,boxes2):
  """
  - torch.tensor 
  -boxes[width,height]

  """
  area = lambda X,Y: X*Y
  boxes1_area = area(boxes1[:,0],boxes1[:,1])
  boxes2_area = area(boxes2[:,0],boxes2[:,1])
  
  width_min = torch.min(boxes1[:,None,0],boxes2[:,0])
  height_min = torch.min(boxes1[:,None,1],boxes2[:,1])
  intersec_area = area(width_min,height_min) 
  union_area = boxes1_area+boxes2_area-intersec_area
  return intersec_area/union_area
def IoU(boxes1,boxes2,format_bboxes="midpoint"):
  """
    bboxes[ x,y,w,g] if midpoint
    bboxes[xmin,ymin,xmax,ymax] if corners
  """
  assert format_bboxes == "corners" or format_bboxes == "midpoint","format_bboxes should be midpoint or corners"
  if format_bboxes== "midpoint":
    Xmin_1 = boxes1[:,0:1] - boxes1[:,2:3]/2
    Xmax_1 = boxes1[:,0:1] + boxes1[:,2:3]/2
    Xmin_2 = boxes2[:,0:1] - boxes2[:,2:3]/2
    Xmax_2 = boxes2[:,0:1] + boxes2[:,2:3]/2

    Ymin_1 = boxes1[:,1:2] - boxes1[:,3:]/2
    Ymax_1 = boxes1[:,1:2] + boxes1[:,3:]/2
    Ymin_2 = boxes2[:,1:2] - boxes2[:,3:]/2
    Ymax_2 = boxes2[:,1:2] + boxes2[:,3:]/2
  else:
    Xmin_1 = boxes1[:,0:1]
    Xmax_1 = boxes1[:,2:3]
    Ymin_1 = boxes1[:,1:2]
    Ymax_1 = boxes1[:,3:]

    Xmin_2 = boxes2[:,0:1]
    Xmax_2 = boxes2[:,2:3]
    Ymin_2 = boxes2[:,1:2]
    Ymax_2 = boxes2[:,3:]
  Xmin =torch.max(Xmin_1[:,None],Xmin_2).squeeze(2)
  Ymin =torch.max(Ymin_1[:,None],Ymin_2).squeeze(2)
  Xmax = torch.min(Xmax_1[:,None],Xmax_2).squeeze(2)
  Ymax = torch.min(Ymax_1[:,None],Ymax_2).squeeze(2)
  area = lambda xmin,ymin,xmax,ymax : (xmax-xmin)*(ymax-ymin)
  boxes1_area = area(Xmin_1,Ymin_1,Xmax_1,Ymax_1)
  boxes2_area = area(Xmin_2,Ymin_2,Xmax_2,Ymax_2)
  intersec_area = area(Xmin,Ymin,Xmax,Ymax)
  union_area = boxes1_area+boxes2_area -intersec_area
  return intersec_area/union_area

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def nms(bboxes,iou_threshold,prob_threshold,format_bboxes="corners"):
  """
  bboxes : list of lists containing all bboxes with [class_pred,prob_pred,xmin,ymin,xmax,ymax]

  """
  assert  type(bboxes)== list, "bboxes should be a list"
  bboxes = [box for box in bboxes if box[1]> prob_threshold]
  bboxes = sorted(bboxes,key = lambda X:X[1], reverse=True)
  bboxes_keep = []
  while bboxes:
    chosen_bboxes = bboxes[0]
    bboxes = [
      box
      for box in bboxes
      if box[0] != chosen_bboxes[0] or intersection_over_union(torch.tensor(box[2:]),torch.tensor(chosen_bboxes[2:]),format_bboxes)<iou_threshold
    ]
    bboxes_keep.append(chosen_bboxes)
  return bboxes_keep

def meanAveragePrecision(pred_bboxes,true_bboxes,iou_threshold=0.5,format_bboxes ="midpoint",num_class=20):
  """
  - pred_bboxes[train_idx,class_pred,prob_pred,x,y,w,h] list
  - true_bboxes[train_idx,class_pred,prob_pred,x,y,w,h] list  
  """
  average_precisions = []
  epsilion = 1e-6
  for c  in range(num_class):
    detections = [pred for pred in pred_bboxes if pred[1]==c]
    ground_truths = [true for true in true_bboxes if true[1]==c]

    # dem so luong grouth truth trong tung buc anh sau do chuyen ve mang 0 nham kiem tra TP 1 lan duy nhat
    # for example: {0:torch.tensor[0,0,0],1:torch.tensor[0,0,0,0,0]}
    amount_grouth = collections.Counter([gt[0] for gt in ground_truths])
    for key,value in amount_grouth.items():
      amount_grouth[key] = torch.zeros(value)
    # sap xep detections theo prob_pred
    detections.sort(key = lambda x: x[2],reverse=True)
    TP = torch.zeros(len(detections))
    FP = torch.zeros(len(detections))
    total_true_boxes = len(ground_truths)
    if total_true_boxes==0:
      continue
    for detection_idx,detection in enumerate(detections):
      # chi lay cac ground_truths trong cung 1 anh voi detection
      ground_truth_img =torch.tensor([gt for gt in ground_truths if gt[0]==detection[0]])
      #tim gia tri lon nhat cua iou giua gt va detection
      num_gti = len(ground_truth_img)
      if num_gti==1:
        ground_truth_img = ground_truth_img.unsqueeze(0)
      iou_total = IoU(torch.tensor([detection[3:]]),ground_truth_img[:,3:],format_bboxes)
      best_iou,best_idx = torch.max(iou_total,dim = 1)
      if best_iou>iou_threshold:
        if amount_grouth[detection[0]][best_idx]==0:
          TP[detection_idx]=1
          amount_grouth[detection[0]][best_idx] = 1
        else:
          FP[detection_idx]=1
      else:
        FP[detection_idx]=1
    TP_cumsum = torch.cumsum(TP,dim = 0)
    FP_cumsum = torch.cumsum(FP,dim = 0)
    recalls = TP_cumsum/(total_true_boxes+epsilion)
    precisions = TP_cumsum/(TP_cumsum+FP_cumsum+epsilion)
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    # torch.trapz for numerical integration
    average_precisions.append(torch.trapz(precisions, recalls))
  print(average_precisions)
  return sum(average_precisions)/len(average_precisions)

def cells2Boxes(predictions,S,anchors,is_preds=True):
  """
  predictions[N,3,S,S,numclasses+5]
  S- the number of cells the image is divided in on the width (height)
  anchors
  is_preds : for prediction or for true boxes2

  out_put :[N,num_anchors*S*S,1+5] with[class_pred,scores,x,y,w,h]
  """
  BATCH_SIZE = predictions.shape[0]
  num_anchors = len(anchors)
  boxes_predictions = predictions[...,1:5]
  if is_preds:
    anchors = anchors.reshape(1,len(anchors),1,1,2)
    boxes_predictions[...,0:2] = torch.sigmoid(boxes_predictions[...,0:2])
    boxes_predictions[...,2:] = anchors*torch.exp(boxes_predictions[...,2:])
    scores =  predictions[...,0:1]
    best_class= torch.argmax(predictions[...,5:],dim = -1).unsqueeze(-1)
  else:
    scores =  predictions[...,0:1]
    best_class = predictions[...,5:6]
  
  cells_indices = (torch.arange(S).repeat(BATCH_SIZE,3,S,1).unsqueeze(-1).to(predictions.device))
  x = 1 / S *(boxes_predictions[...,0:1]+cells_indices)
  y = 1 / S *(boxes_predictions[...,1:2]+cells_indices.permute(0,1,3,2,4))
  w_h = 1 / S *boxes_predictions[...,2:4]
  output_boxes = torch.cat((best_class,scores,x,y,w_h), dim = -1).reshape(BATCH_SIZE,num_anchors*S*S,6)
  return output_boxes.tolist()
def GetPredandTrueBoxes(
  model,
  data_loader,
  iou_threshold,
  anchors,
  threshold,
  box_format = "midpoint",
  device = "cuda"):
  """
  return [[train_idx,class_pred,score,x,y,w,h]]
  """
  model.eval()
  all_pred_boxes=[]
  all_true_boxes=[]
  train_idx = 0
  for X,y in tqdm(data_loader):
    X = X.to(device)
    with torch.no_grad():
      predictions = model(X)
    bboxes = [[] for _ in range(config.BATCH_SIZE)]
    for i in range(3):
      S = predictions[i].shape[2]
      anchor = torch.tensor([*anchors[i]]).to(device)*S
      boxes_scaler_i = cells2Boxes(predictions[i],S,anchor)

      for idx,(box) in enumerate(boxes_scaler_i):
        bboxes[idx]+=box
    true_bboxes = cells2Boxes(y[2], S,anchor,is_preds = False)

    for idx in range(config.BATCH_SIZE):
      nms_boxes = nms(bboxes[idx],iou_threshold = iou_threshold,prob_threshold=threshold,format_bboxes=box_format)
      for nms_box in nms_boxes:
        all_pred_boxes.append([train_idx]+nms_box)
      for box in true_bboxes:
        all_true_boxes.append([train_idx]+box)
      train_idx+=1
  model.train()
  return all_pred_boxes,all_true_boxes


def checkClassAccuracy(model,loader,threshold):
  model.eval()
  total_class_pred,correct_class = 0,0
  total_obj_pred,correct_obj = 0,0
  total_noobj_pred,correct_noobj =0,0
  for (x,y) in tqdm(loader):
    x = x.to(config.DEVICE)
    with torch.no_grad():
      out = model(x)
    for i in range(3):
      y[i]=y[i].to(config.DEVICE)

      obj = y[i][...,0]==1
      noobj = y[i][...,0]==0

      correct_class += torch.sum(torch.argmax(out[i][...,5:][obj],dim = -1)== y[i][...,5][obj])
      total_class_pred +=torch.sum(obj)
      obj_pred = torch.sigmoid(out[i][...,0])>threshold
      correct_obj += torch.sum(obj_pred[obj] == y[i][...,0][obj])
      total_obj_pred += torch.sum(obj)

      correct_noobj += torch.sum(obj_pred[noobj]== y[i][...,0][noobj])
      total_noobj_pred += torch.sum(noobj)
  
  print(f"Class accuracy is: {(correct_class/(total_class_pred+1e-16))*100:2f}%")
  print(f"No obj accuracy is: {(correct_noobj/(total_noobj_pred+1e-16))*100:2f}%")
  print(f"Obj accuracy is: {(correct_obj/(total_obj_pred+1e-16))*100:2f}%")
  model.train()

def LoadCheckPoint(checkpoint_file,model,optimizer,lr):
  print("=> Loading checkpoint")
  checkpoint = torch.load(checkpoint_file,map_location=config.DEVICE)
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])

  for param_group in optimizer.param_groups:
    param_group["lr"] = lr