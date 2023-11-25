import cv2
import numpy as np
import torch
import albumentations as A

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32') # original is uint16
    img = img / 255
    # img = A.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_msk(path):
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    # msk /= 255.0
    return msk

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, msk_paths=[], transforms=None, dataset_len=None, cache=False):
        self.img_paths  = img_paths
        self.msk_paths  = msk_paths
        self.transforms = transforms
        self.dataset_len = dataset_len
        self.len_img_paths = len(self.img_paths)
        self.len_msk_paths = len(self.msk_paths)
        self.cache = cache
        if cache:
            self.cache_images()
            self.cache_masks()
        
    def __len__(self):
        if self.dataset_len:
            return self.dataset_len
        return len(self.img_paths)
    
    def cache_images(self):
        self.images = {i:load_img(i) for i in self.img_paths}
    
    def cache_masks(self):
        self.masks = {i:load_msk(i) for i in self.msk_paths}
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index%self.len_img_paths]
        
        if self.cache:
            img = self.images[img_path]
        else:
            img = load_img(img_path)
        
        if len(self.msk_paths)>0:
            msk_path = self.msk_paths[index%self.len_msk_paths]
            if self.cache:
                msk = self.masks[msk_path]
            else:
                msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            # img = np.expand_dims(img, 0)
            return torch.tensor(img), torch.tensor(msk)
        else:
            orig_size = img.shape
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            # img = np.expand_dims(img, 0)
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(np.array([orig_size[0], orig_size[1]]))