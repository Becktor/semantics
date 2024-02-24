#!/usr/bin/env python3
import torch
import torch.nn as nn
import wandb
import os
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from src.features.tensor_utils import Resize

class DinoFeaturizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg['dim']
        self.pretrained = cfg['pretrained']
        self.model = self.load_dino_model()  # Load DINO model
        self.dropout = nn.Dropout2d(p=0.1)
        self.resize = None
        self.n_feats = 384 if cfg['model'] == "vit_small" else 768
        self.conv_features = self.create_conv_features()  # Create conv features from dino
        self.img_conv_features = self.create_img_conv_features()  # Create conv features from img
        self.pred = self.create_pred_head()  # Create prediction head
        self.lossfn = nn.CrossEntropyLoss(reduction='mean')

    def load_dino_model(self):
        model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14')  # Load DINO model from hub
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    
    def create_conv_features(self):
        conv_features = nn.Sequential(
            nn.Conv2d(self.n_feats, 128, 1),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(128, 48, 3),
            self.dropout,
            nn.ReLU(),
        )
        if self.pretrained:
           md = torch.load("trained_models/segmentation.pth")  # Load pretrained chicane head model
           conv_features.load_state_dict(md, strict=False)
        return conv_features
   
    def create_img_conv_features(self):
        img_conv_features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding='same'),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding='same'),
            self.dropout,
            nn.ReLU(),
        )
        if self.pretrained:
           md = torch.load("trained_models/segmentation.pth")  # Load pretrained chicane head model
           img_conv_features.load_state_dict(md, strict=False)
        return img_conv_features

    def create_pred_head(self):
        pred_head = nn.Sequential(
            nn.Conv2d(64, 10, 3),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(10, 25, 3),
        )
        if self.pretrained:
           md = torch.load("trained_models/segmentation.pth")  # Load pretrained prediction head model
           pred_head.load_state_dict(md, strict=False)
        return pred_head

    def loss(self, x ,y):
        los = self.lossfn(x,y)
        return los

    def predict(self, x):
        x = self.forward(x)
        softmx = torch.softmax(x, dim=1)
        return softmx
    
    def forward(self, img):
        if self.resize is None:
            self.resize = Resize((img.shape[2]+4, img.shape[3]+4))
        self.img_size = img.shape[2::]
        # Forward pass of DINO model
        dino_features = self.forward_dino(img)
        conv_features = self.conv_features(dino_features)
        # Resize conv features to match img size
        resized_conv_features = self.resize(conv_features)
        resized_img = self.resize(img)
        # Forward pass of img conv features
        img_features = self.img_conv_features(resized_img)
        # Concatenate conv features and img features
        combined_features = torch.cat([resized_conv_features, img_features], dim=1)
        # Forward pass of prediction head
        predicted_output = self.pred(combined_features)
        return predicted_output, dino_features

    def forward_dino(self, img):
        with torch.no_grad():
            assert img.shape[2] % self.cfg['patch_size'] == 0
            assert img.shape[3] % self.cfg['patch_size'] == 0
            feat = self.model.get_intermediate_layers(img, n=1, reshape=True)[0]  # Get intermediate layers from DINO model
            return feat
