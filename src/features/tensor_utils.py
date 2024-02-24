import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

def tensor_to_PIL(tensor):
    return Image.fromarray(tensor.astype(np.uint8))

def cutmix(batch, alpha):
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets[:,:, y0:y1, x0:x1] = shuffled_targets[:,:, y0:y1, x0:x1]
    return data, targets

class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def forward(self, x):
        return torch.nn.functional.interpolate(x, self.size, mode='bilinear', align_corners=False)

def plot_tensor(tensor, x, log_dict=None):
    # ensure input tensor shape
    tshape=tensor.shape
    #assert tensor.shape == (384, 16, 22), "input tensor shape must be [384, 14, 14]"
    
    # reshape tensor to have 3 channels for each of the first 9 images
    images = tensor.view(-1, 3, tshape[-2],tshape[-1])[:9]  # take only the first 9 sets of 3 channels
    
    fig, axs = plt.subplots(3, 3, figsize=(6, 6))
    
    for idx, img in enumerate(images):
        i, j = divmod(idx, 3)
        
        # rearrange tensor shape for plotting
        img = img.permute(1, 2, 0).cpu().detach().numpy()
        
        # normalize image for visualization (if necessary)
        img = (img - img.min()) / (img.max() - img.min())
        
        axs[i, j].imshow(img)
        axs[i, j].axis('off')
    
    plt.tight_layout()
    log_dict[f"features_{x}"]= fig
    plt.close(fig)


class JointTransform:
    def __init__(self,h,w, n_labels, lbl_to_idx):
        self.n_labels = n_labels
        self.lbl_to_idx = lbl_to_idx
        # spatial transforms
        self.spatial_transforms = transforms.Compose([
            transforms.Resize((int(h*1.2), int(w*1.2))),
            transforms.RandomResizedCrop(size=(h, w), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.AugMix(),
            transforms.ToTensor(),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((int(h*1.2), int(w*1.2))),
            transforms.RandomResizedCrop(size=(h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.AugMix(all_ops=False, mixture_width=1, chain_depth=1),
            transforms.ToTensor(),
        ])

        self.ToTensor = transforms.ToTensor()
        self.val_transforms = transforms.Compose([transforms.Resize((h,w)),
                                             transforms.ToTensor(),])
    
    def set_augmix_severity(self, severity):
        self.spatial_transforms.transforms[-2].severity = severity
        self.mask_transforms.transforms[-2].severity = severity

    def __call__(self, image, mask, train=True):
        # apply spatial transforms
        seed = torch.randint(0, 2**32, ())
        torch.manual_seed(seed)  # ensure same transform for image and mask
        if(train):
            image_t = self.spatial_transforms(image)
        else:
            image_t = self.val_transforms(image)
        # separate each grayscale channel into a separate channel
        mask = np.array(mask)
        uniq = np.unique(mask)
        mask_t = np.zeros((self.n_labels, image_t.shape[1], image_t.shape[2]))
        for n in uniq:
            if n == 0:
                continue
            torch.manual_seed(seed)
            channel_n = mask == n
            channel_n = Image.fromarray(channel_n.astype(np.uint8)*255)
            if(train):
                mask_t_n = self.mask_transforms(channel_n)
            else:
                mask_t_n = self.val_transforms(channel_n)
            mask_t_n[mask_t_n>1]=1
            mask_t[self.lbl_to_idx[n],:,:] = mask_t_n
        # last mask_t is the background
        mask_t[self.n_labels-1,:,:] = 1 - mask_t.sum(0)
        return image_t, mask_t    

def plot_tensor_image(img, name):
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.savefig(name)

def vectorize_tensor(o_m, idx_to_lbl):
    #invert dict
    #idx_to_lbl = {v: k for k, v in lbl_to_idx.items()}
    o_m = o_m.cpu().numpy()
    o_m_m = np.vectorize(idx_to_lbl.get)(o_m)
    #print(np.unique(o_m))
    return o_m_m

def vectorize_tensor_rgb(vector, idx_to_rgb):
    #invert dict
    #idx_to_lbl = {v: k for k, v in lbl_to_idx.items()}
    vector = vector.cpu().numpy()
    idx_to_rgb = [
        {k:v[0] for k,v in idx_to_rgb.items()},
        {k:v[1] for k,v in idx_to_rgb.items()},
        {k:v[2] for k,v in idx_to_rgb.items()}
        ]
    rgb = np.zeros((vector.shape[0], vector.shape[1],vector.shape[2], 3))
    for i, color_map in enumerate(idx_to_rgb):
        vectorized = np.vectorize(color_map.get)(vector)
        rgb[:,:,:,i] = vectorized

    return rgb