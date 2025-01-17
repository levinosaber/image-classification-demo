import sys

import torch
from tqdm import tqdm

from utils.distribute_utils import  is_main_process, reduce_value   
from utils.lr_methods import warmup 


def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=False, lr_method=None):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    train_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)

    optimizer.zero_grad()
    
    lr_scheduler = None
    if epoch == 0  and lr_method == warmup : 
        warmup_factor = 1.0/1000
        warmup_iters = min(1000, len(data_loader) -1)

        lr_scheduler = warmup(optimizer, warmup_iters, warmup_factor)
    
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    
    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred = model(images.to(device))
            loss = loss_function(pred, labels.to(device))

            pred_class = torch.max(pred, dim=1)[1]
            acc_num += torch.eq(pred_class, labels.to(device)).sum()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += reduce_value(loss, average=True).detach()

        if is_main_process():
            info = '[epoch{}]: learning_rate:{:.5f}'.format(
                epoch + 1, 
                optimizer.param_groups[0]["lr"]
            )
            data_loader.desc = info 
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  
            lr_scheduler.step()

    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    
    return train_loss.item() / (step + 1), acc_num.item() / sample_num
        
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    num_samples = len(data_loader.dataset) 

    sum_num = torch.zeros(1).to(device)
 
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred_class = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred_class, labels.to(device)).sum()

    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    
    sum_num = reduce_value(sum_num, average=False)
    val_acc = sum_num.item() / num_samples

    return val_acc