import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import segmentation_models_pytorch as smp
from training_utils import fetch_scheduler
from train import train_one_epoch
from validate import valid_one_epoch
from collections import defaultdict

from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

from dataset import BuildDataset
from utils import set_seed
from config import CFG
import wandb
import numpy as np
import time
import copy
import gc



def build_model(backbone, num_classes, device):
    model = smp.Unet(
        encoder_name=backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=CFG.pretrained,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(device)
    return model

def run_training(model, optimizer, scheduler, num_epochs):    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric      = -np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(
            CFG, model, optimizer, scheduler, 
            dataloader=train_loader, 
            device=CFG.device, epoch=epoch
        )
        
        if epoch%CFG.eval_every==0:
            val_loss, val_scores = valid_one_epoch(
                model, valid_loader, 
                device=CFG.device,
                optimizer=optimizer
            )

            # scheduler.step(val_loss)
            val_dice, val_jaccard = val_scores
            history['Train Loss'].append(train_loss)
            history['Valid Loss'].append(val_loss)
            history['Valid Dice'].append(val_dice)
            history['Valid Jaccard'].append(val_jaccard) 
            
            wandb.log({'epoch_train_loss': train_loss})
            wandb.log({'epoch_val_loss': val_loss})
            wandb.log({'epoch_val_dice': val_dice})
            # wandb.log({'epoch_val_jaccard': val_jaccard})
            
            print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
            print(f'Valid Loss: {val_loss}')
            
            # deep copy the model
            if val_jaccard > best_metric:
                print(f"{c_}Valid loss Improved ({best_metric} ---> {val_jaccard})")
                best_dice    = val_dice
                best_jaccard = val_jaccard
                best_metric = val_jaccard
                best_epoch   = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"{CFG.output_dir}/best_epoch.bin"
                torch.save(model.state_dict(), PATH)
                print(f"Model Saved{sr_}")
                
            last_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{CFG.output_dir}/last_epoch.bin"
            torch.save(model.state_dict(), PATH)
                
            print(); print()
        
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_metric))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

if __name__=='__main__':
    cfg_dict = dict(CFG.__dict__)
    cfg_dict.pop('__weakref__')
    cfg_dict.pop('__dict__')
    cfg_dict.pop('device')
    cfg_dict.pop('data_transforms')

    os.makedirs(CFG.output_dir, exist_ok=True)
    
    wandb.init(
        project="building_segmentations",
        config=cfg_dict
    )
    set_seed(CFG.seed)

    gt_df = pd.read_csv(CFG.gt_df)
    gt_df["img_path"] = gt_df["img_path"].apply(lambda x: os.path.join(CFG.data_root, x))
    gt_df["mask_path"] = gt_df["mask_path"].apply(lambda x: os.path.join(CFG.data_root, x))
    train_df = gt_df[gt_df['split']=='train'].reset_index(drop=True)
    valid_df = gt_df[gt_df['split']=='valid'].reset_index(drop=True)
    train_img_paths = train_df["img_path"].values.tolist()
    train_msk_paths = train_df["mask_path"].values.tolist()
    valid_img_paths = valid_df["img_path"].values.tolist()
    valid_msk_paths = valid_df["mask_path"].values.tolist()
    if CFG.debug:
        train_img_paths = train_img_paths[:CFG.train_bs*5]
        train_msk_paths = train_msk_paths[:CFG.train_bs*5]
        valid_img_paths = valid_img_paths[:CFG.valid_bs*3]
        valid_msk_paths = valid_msk_paths[:CFG.valid_bs*3]

    train_dataset = BuildDataset(train_img_paths, train_msk_paths, transforms=CFG.data_transforms['train'], dataset_len=CFG.dataset_len, cache=CFG.cache)
    valid_dataset = BuildDataset(valid_img_paths, valid_msk_paths, transforms=CFG.data_transforms['valid'])
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=8, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=8, shuffle=False, pin_memory=True)

    CFG.T_max = ((len(train_loader) // CFG.n_accumulate) * CFG.epochs) + 100
    model = build_model(CFG.backbone, CFG.num_classes, CFG.device)
    model.load_state_dict(torch.load('/mnt/SSD/workspace/roads_buildings/src/exps/1700875882.2132142/best_epoch.bin'))
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(CFG, optimizer)

    model, history = run_training(
        model, optimizer, scheduler,
        num_epochs=CFG.epochs
    )