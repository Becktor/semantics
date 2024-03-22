import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import v2

# Define the RUGD dataset class
class RUGD(Dataset):
    # Initialize the dataset
    def __init__(self, labels_dir, transform=None, transform_t=None, train=True, custom_transform=None):
        self.labels_dir = labels_dir  # Directory of labels
        self.transform = transform  # Transform for images
        self.target_transform = transform_t  # Transform for labels
        self.train = train  # Whether the dataset is for training or not
        self.lbl_to_idx = {}  # Mapping from label to index
        # Get the list of image and label files
        label_list = sorted(glob.glob(os.path.join(self.labels_dir, "*/*.png"), recursive=True))
        # Path to the colormap file
        colormap_path = os.path.join(self.labels_dir, "RUGD_annotation-colormap.txt")
        self.colormap = {}  # Mapping from index to color
        # Load the label to index mapping
        self.load_label_to_idx(colormap_path)
        # Sorted lists of image and label files
        self.sorted_label_list = label_list
        self.custom_transform = custom_transform  # Custom transform

    # Load the label to index mapping from a file
    def load_label_to_idx(self, path):
        with open(path, 'r') as f:
            for line in f:
                idx, label, r, g, b = line.split()
                self.lbl_to_idx[int(idx)] = label
                self.colormap[int(idx)] = (int(r), int(g), int(b))

    # Get the length of the dataset
    def __len__(self):
        return len(self.sorted_label_list)

    # Get an item from the dataset
    def __getitem__(self, idx):
        lbl_path = self.sorted_label_list[idx]
        img_path = lbl_path.replace("orig", "imgs")
        pil_image = Image.open(img_path)
        label = Image.open(lbl_path)

        if self.train:
            if self.custom_transform:
                image, label = self.custom_transform(image, label)
            else:
                image = self.transform(pil_image)
                
                label = self.target_transform(label)
                label[label == 255] = 0
                label = F.one_hot((label[0]), 25)
                label = label.permute(2, 0, 1).float()

        return image, label

# Main function
def main(labels_dir, img_dir):
    # Define the image size and the transform
    h=550
    nh = (550//14)*14
    w=688
    nw = (688//14)*14

    transform = transforms.Compose([

        v2.CenterCrop((int(nh), int(nw))),
        v2.ToDtype(torch.float32, scale=True)
  
    ])
    label_tranform = transforms.Compose([
        
        v2.CenterCrop((int(nh), int(nw))),
        v2.ToDtype(torch.long, scale=False)
 
    ])

    # Create the dataset
    dataset = RUGD(labels_dir=labels_dir, transform=transform,label_tranform=label_tranform)
    # Get the first item from the dataset
    a,b = dataset.__getitem__(0)

    # Print the shapes of the image and the label
    print(a.shape)
    print(b.shape)

# If this file is run directly, parse command-line arguments and run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RUGD dataset.')
    parser.add_argument('--labels_dir', type=str, 
                        required=False, help='Path to the labels directory', 
                        default='/mnt/d/Data/RUGD/orig')
    parser.add_argument('--img_dir', type=str, 
                        required=False, help='Path to the image directory', 
                        default='/mnt/d/Data/RUGD/imgs')
    args = parser.parse_args()
    if args is not None:
        main(args.labels_dir, args.img_dir)
    else:
        main('/mnt/d/Data/RUGD/orig', '/mnt/d/Data/RUGD/imgs')