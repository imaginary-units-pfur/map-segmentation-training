import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import wandb
import gc
from training_utils import criterion
from torch.cuda import amp

def train_one_epoch(CFG, model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    print()
    print(f"lr = {optimizer.param_groups[0]['lr']}")
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ', dynamic_ncols=True)
    for step, (images, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(CFG.loss_func, y_pred, masks)
            wandb.log({'train_loss': loss.item()})
            loss   = loss / CFG.n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({'lr': current_lr})
        pbar.set_postfix( epoch=f'{epoch}',
                          train_loss=f'{epoch_loss:0.4f}',
                          lr=f'{current_lr:0.5f}')
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss