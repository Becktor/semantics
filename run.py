#!/usr/bin/env python3
import torch
import torch.nn as nn
import wandb
import os
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from data.RUGD_dl import RUGD
from features.tensor_utils import *
from architectures.dino import DinoFeaturizer
from models.train_model import ModelTrainer

# Constants
NUM_EPOCHS = 200
BATCH_SIZE = 40
DATA_ROOT = r"/mnt/d/Data"
OUTPUT_DIR = r"reports/output/100_aug_mix_adding_images_back"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = {'dim': 4, 'patch_size': 14, 'model': "vit_small", 'pretrained': False}
    model = DinoFeaturizer(cfg).to(DEVICE)
    #transform = JointTransform(224, 308, 24, lbl_to_idx)
    h=550
    nh = (550//14)*14
    w=688
    nw = (688//14)*14

    transform = transforms.Compose([

        transforms.CenterCrop((int(nh), int(nw))),
        transforms.ToTensor(),
        #transforms.RandomResizedCrop(size=(14*nh, 14*nw)),
        #transforms.RandomHorizontalFlip(p=0.5),
    ])
    transform_t = transforms.Compose([
        transforms.CenterCrop((int(nh), int(nw))),
    ])
    model_trainer = ModelTrainer(model, DATA_ROOT, transform,transform_t, BATCH_SIZE, NUM_EPOCHS, DEVICE)

    model_trainer.train_model(transform)


if __name__ == '__main__':
    wandb.init(project="terrain_classifier")
    main()