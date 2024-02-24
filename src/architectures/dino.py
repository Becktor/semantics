#!/usr/bin/env python3
import torch
import torch.nn as nn
import wandb
import os
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from features.tensor_utils import Resize

# label ids
class DinoFeaturizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg['dim']
        self.pretrained = cfg['pretrained']
        self.model = self.load_dino_model()
        self.dropout = nn.Dropout2d(p=0.1)
        self.resize = None
        self.n_feats = 384 if cfg['model'] == "vit_small" else 768
        self.chicane_head = self.create_chicane_head()
        self.pred = self.create_pred_head()
        self.lossfn = nn.CrossEntropyLoss(reduction='mean')

    def load_dino_model(self):
        model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14')
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def create_chicane_head(self):
        chicane_head = nn.Sequential(
            nn.Conv2d(self.n_feats, 128, 1),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(128, 64, 3),
            self.dropout,
            nn.ReLU(),
        )
        if self.pretrained:
           md = torch.load("trained_models/segmentation.pth")
           chicane_head.load_state_dict(md, strict=False)
        return chicane_head

    def create_pred_head(self):
        pred_head = nn.Sequential(
            nn.Conv2d(67, 10, 3),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(10, 25, 3),
        )
        if self.pretrained:
           md = torch.load("trained_models/segmentation.pth")
           pred_head.load_state_dict(md, strict=False)
        return pred_head

    def loss(self, x ,y):
        los = self.lossfn(x,y)
        return los

    def predict(self, x):
        x = self.forward(x)
        softmx = torch.softmax(x, dim=1)
        return softmx
    
    def forward(self, img, return_class_feat=False):
        if self.resize is None:
            self.resize = Resize((img.shape[2]+4, img.shape[3]+4))
        self.img_size = img.shape[2::]
        features = self.forward_dino(img)
        chicane = self.chicane_head(features)
        rsh = self.resize(chicane)
        rimg=self.resize(img)
        #concat image and chicane
        chicane_rgb = torch.cat([rsh, rimg], dim=1)
        chicane_rgb = self.pred(chicane_rgb)
        return chicane_rgb, features

    def forward_dino(self, img, return_class_feat=False):
        """
        Forward pass of the model.
        """
        with torch.no_grad():
            assert img.shape[2] % self.cfg['patch_size'] == 0
            assert img.shape[3] % self.cfg['patch_size'] == 0

            # get selected layer activations
            feat = self.model.get_intermediate_layers(img, n=1, reshape=True)[0] # returns last features
            return feat
