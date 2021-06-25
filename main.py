from dataset import YOLODataset,LoadData
from loss import YOLOLoss
from train import train_epoch
from tqdm import tqdm
from model import YOLOv3
import config
import utils
import torch
def main():
  train_iter,val_iter = LoadData("PASCAL_VOC/50examples.csv","PASCAL_VOC/2examples.csv")

  torch.cuda.empty_cache()
  loss_fn = YOLOLoss()
  net = YOLOv3()
  net = net.to(config.DEVICE)
  scaler = torch.cuda.amp.GradScaler()
  optimizer = torch.optim.Adam(
      net.parameters(),
      lr = config.LEARNING_RATE,
      weight_decay = config.WEIGHT_DECAY)
  # if config.LOAD_MODEL:
  #     utils.LoadCheckPoint(
  #         config.CHECKPOINT_FILE,net,optimizer,config.LEARNING_RATE
  #     )
  scaler_anchors = (
      torch.tensor(config.ANCHORS)
      * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)).to(config.DEVICE)

  for epoch in range(config.NUM_EPOCHS):
    train_epoch(train_iter,net,loss_fn,optimizer,scaler,scaler_anchors)
    if epoch>0 and epoch %3==0:
        utils.checkClassAccuracy(net,val_iter,config.CONF_THRESHOLD)
        pred_boxes,true_boxes = utils.GetPredandTrueBoxes(
                                net,val_iter,
                                config.NMS_IOU_THRESH,
                                config.ANCHORS,
                                config.CONF_THRESHOLD
                                )
        mapval = utils.meanAveragePrecision(
                                    pred_boxes,true_boxes,
                                    config.MAP_IOU_THRESH,
                                    format_bboxes = "midpoint",
                                    )
        print(f"Map :{mapval.item()}")
        net.train()

if __name__ == "__main__":
  main()