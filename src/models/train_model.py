#!/usr/bin/env python3
import torch
import torch.nn as nn
import wandb
import os
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from src.data.RUGD_dl import RUGD
from src.features.tensor_utils import *
from src.architectures.dino import DinoFeaturizer
from torchvision.transforms import v2

# Constants
NUM_EPOCHS = 200
BATCH_SIZE = 40
DATA_ROOT = r"/mnt/d/Data"
OUTPUT_DIR = r"reports/output/100_aug_mix_adding_images_back"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ModelTrainer class
class ModelTrainer():
    """
    ModelTrainer class for training and evaluating a model.
    """
    def __init__(self, model, data_root, transform, transform_t, batch_size, num_epochs, device):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler(model)
        self.best_avg_loss = float('inf')
        self.epoch = 0
        self.data_handler = DataHandler(data_root, transform, transform_t, batch_size, num_epochs, device)
        self.train_loader = self.data_handler.train_loader
        self.test_loader = self.data_handler.test_loader
        self.lbl_to_idx = self.train_loader.dataset.dataset.colormap
    
    def setup_optimizer_and_scheduler(self, model):
        # Setup the optimizer and scheduler for training
        params = list(model.conv_features.parameters()) +list(model.img_conv_features.parameters()) + list(model.pred.parameters())
        optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.80)
        return optimizer, scheduler
    
    def adjust_training_parameters(self, epoch, joint_transforms):
        # Adjust training parameters based on the current epoch
        cutmix_val = None
        if epoch == 0:
            cutmix_val = 0.5
            #joint_transforms.set_augmix_severity(3)
        elif epoch == int(NUM_EPOCHS*.5):
            cutmix_val = 0.75
            #joint_transforms.set_augmix_severity(2)
        elif epoch == int(NUM_EPOCHS*.8):
            cutmix_val = 0.8
            #joint_transforms.set_augmix_severity(1)
        return cutmix_val

    def train_step(self, inputs, masks):
        # Perform a single training step
        self.optimizer.zero_grad()
        outputs, _ = self.model(inputs)
        loss = self.model.loss(outputs, masks)
        loss.backward()
        self.optimizer.step()
        return loss.item(), outputs

    def train_model(self, joint_transforms):
        # Train the model for the specified number of epochs
        for epoch in range(self.num_epochs):
            self.model.train()
            cutmix_val = self.adjust_training_parameters(epoch, joint_transforms)
            tqdm_loader = tqdm(self.train_loader)
            net_l = []
            for i, data in enumerate(tqdm_loader):
                inputs, masks = self.data_handler.preprocess_data(data, cutmix_val)
                loss, outputs = self.train_step(inputs, masks)
                net_l.append(loss)
                tqdm_loader.set_description(f"e:{epoch} lr:{self.optimizer.param_groups[0]['lr']} -- mean Loss: {np.mean(net_l):.4f} currloss: {loss:.4f} ")
                if i % 10 == 0 and epoch % 5 == 0:
                    masks_o = vectorize_tensor_rgb(masks.argmax(1), self.lbl_to_idx)
                    outputs_o = vectorize_tensor_rgb(outputs.argmax(1), self.lbl_to_idx)
                    wandb.log({ 
                        "train-input": [wandb.Image(inputs[j]) for j in range(2)],
                        "train-masks": [wandb.Image((tensor_to_PIL(masks_o[j]))) for j in range(2)],
                        "train-output": [wandb.Image((tensor_to_PIL(outputs_o[j]))) for j in range(2)]
                    })
        
            self.scheduler.step()
            if epoch % 5 == 0:
                self.evaluate_model()

    def evaluate_model(self):
        # Evaluate the model on the test set
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_precision = 0.0
        num_batches = len(self.test_loader)

        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs, features = self.model(images)
                loss = self.model.loss(outputs, masks)

                total_loss += loss.item()
                precision = torch.mean((torch.argmax(masks, axis=1) == torch.argmax(outputs, axis=1)).float()).item()
                total_precision += precision

        avg_loss = total_loss / num_batches
        avg_precision = total_precision / num_batches
        self.log_and_save(images, masks, outputs, features, avg_loss, avg_precision)
        return avg_loss, avg_precision, features, images, masks

    def log_and_save(self, images, masks, outputs, features, avg_loss, avg_precision):
        # Log and save the model and evaluation results
        o_m = vectorize_tensor_rgb(masks.argmax(1), self.lbl_to_idx)
        o_o = vectorize_tensor_rgb(outputs.argmax(1), self.lbl_to_idx)
        a_range = range(0, 20, 4)
        log_dict = {
            "images": [wandb.Image(images[i]) for i in a_range],
            "masks_0": [wandb.Image(tensor_to_PIL(o_m[i])) for i in a_range],
            "outputs": [wandb.Image(tensor_to_PIL(o_o[i])) for i in a_range]
        }

        wandb.log(log_dict)
        wandb.log({
            "avg_loss": np.mean(avg_loss),
            "avg_precision": np.mean(avg_precision)
        })
        
        torch.save(self.model.state_dict(), 'mm_model_latest.pth'.format(self.epoch))
        avg_loss_mean = np.mean(avg_loss)
        if avg_loss_mean < self.best_avg_loss:
            self.best_avg_loss = avg_loss_mean 
            torch.save(self.model.state_dict(), 'mm_model_best.pth'.format(self.epoch))

class DataHandler():
    def __init__(self, data_root, transform,transform_t, batch_size, num_epochs, device):
        self.data_root = data_root
        self.transform = transform
        self.transform_t = transform_t
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.create_data_loaders()
        


    def preprocess_data(self, data, cutmix_val):
        # Preprocess the input data
        inputs, masks = data
        if cutmix_val is not None and np.random.rand() < cutmix_val:
            inputs, masks = cutmix(data, 1)
        return inputs.to(self.device), masks.to(self.device)

    def load_datasets(self):
        # Load the training and validation datasets
        data_set = RUGD(
            labels_dir=os.path.join(self.data_root,"RUGD","orig"), 
            transform=self.transform,
            transform_t=self.transform_t,
        )
    
        torch.manual_seed(0)  # Set seed for reproducibility
        train_split = int(0.9 * len(data_set))
        test_split = len(data_set) - train_split
        train_set, val_set = random_split(data_set, [train_split, test_split])
        return train_set, val_set

    def create_data_loaders(self):
        train_set, val_set = self.load_datasets()
        # Create data loaders for training and testing
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(val_set, batch_size=20, shuffle=False, drop_last=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = {'dim': 4, 'patch_size': 14, 'model': "vit_small", 'pretrained': False}
    model = DinoFeaturizer(cfg).to(DEVICE)
    #transform = JointTransform(224, 308, 24, lbl_to_idx)
    h=550
    nh = (550//14)*14
    w=688
    nw = (688//14)*14

    transform = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop((int(nh), int(nw))),
        v2.ToDtype(torch.float32, scale=True)
  
    ])
    label_tranform = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop((int(nh), int(nw))),
        v2.ToDtype(torch.long, scale=False)
 
    ])
    model_trainer = ModelTrainer(model, DATA_ROOT, transform, label_tranform, BATCH_SIZE, NUM_EPOCHS, DEVICE)

    model_trainer.train_model(transform)


if __name__ == '__main__':
    wandb.init(project="panoptic-seg")
    main()
