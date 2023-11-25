import torch
import cv2
import albumentations as A
import time

class CFG:
    seed          = 42
    debug         = False # set debug=False for Full Training
    exp_name      = 'baseline'
    comment       = None
    output_dir    = f'./exps/{time.time()}'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b7'
    pretrained    = 'imagenet'
    train_bs      = 2
    valid_bs      = 2
    img_size      = [1024, 1024]
    epochs        = 100
    num_patience  = 5
    # n_accumulate  = max(1, 64//train_bs)
    n_accumulate  = 1
    lr            = 0.002
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(19/(train_bs*n_accumulate)*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_fold        = 5
    num_classes   = 1
    eval_every    = 1
    dataset_len   = 500
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache         = True

    data_root = "/mnt/SSD/workspace/roads_buildings/"
    # gt_df = data_root+"satellite-building-segmentation//df.csv"
    # gt_df = data_root+"df_stitched.csv"
    gt_df = data_root+"df.csv"
    # gt_df = data_root+"df_pretrained.csv"
    # train_groups = ['kidney_1_dense', 'kidney_1_voi', "kidney_2", "kidney_3_dense", 'kidney_3_sparse']
    # train_groups = ['kidney_1_dense', "kidney_2"]
    # train_groups = ["kidney_1_dense"]
    # valid_groups = ['kidney_2']
    # valid_groups = ['kidney_3_dense']
    loss_func     = "DiceLoss"

    data_transforms = {
        "train": A.Compose([
            # A.Resize(*img_size, interpolation=cv2.INTER_NEAREST, p=1),
            A.OneOf([
                A.Resize(*img_size, interpolation=cv2.INTER_NEAREST, p=0.5),
                A.Compose([
                    A.PadIfNeeded(*img_size, p=1),
                    # A.CropNonEmptyMaskIfExists(*img_size, p=1),
                    A.RandomCrop(*img_size, p=1)
                    # A.OneOf([
                    # ])
                ], p=0.8),
            ], p=1),
            A.RandomBrightnessContrast(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(max_height=32, max_width=32, min_holes=1, max_holes=16, p=0.5),
            A.ElasticTransform(p=0.5),
            A.GaussianBlur(p=0.5),
            A.InvertImg(p=0.5),
        ], p=1.0),
        
        "valid": A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }