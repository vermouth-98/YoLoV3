import torch
import utils
from tqdm import tqdm
import config
import utils
torch.backends.cudnn.benchmark = True

def train_epoch(train_iter,net,loss_fn,optimizer,scaler,scaler_anchors):
  net.train()
  loop = tqdm(train_iter,leave = True)
  losses =[]
  for batch_idx,(X,target) in enumerate(loop):
    
    X= X.to(config.DEVICE)
    target_0,target_1,target_2 = (
      target[0].to(config.DEVICE),
      target[1].to(config.DEVICE),
      target[2].to(config.DEVICE))

    with torch.cuda.amp.autocast():
      predictions = net(X)
      l = (loss_fn(predictions[0],target_0,scaler_anchors[0])
          + loss_fn(predictions[1],target_1,scaler_anchors[1])
          + loss_fn(predictions[2],target_2,scaler_anchors[2]))
    losses.append(l.item())
    optimizer.zero_grad()
    scaler.scale(l).backward()
    scaler.step(optimizer)
    scaler.update()

    #update progress bar
    mean_loss = sum(losses)/len(losses)
    loop.set_postfix(loss = mean_loss)


