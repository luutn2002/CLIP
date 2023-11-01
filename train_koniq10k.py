from models.maple_iqa import build_mapleiqa_v5
from torch_utils.dataset import CustomKonIQ10kDataset
from torch_utils.metrics import srocc, plcc

from torch.utils.data import DataLoader, random_split
from pandas import read_csv
from numpy import where, unique, apply_along_axis, inf
from torch.cuda import is_available
from torch import Tensor, no_grad, save, load, isnan
from torch.nn import MSELoss
from torch.optim import Adam, SGD, AdamW
from torch.autograd import set_detect_anomaly
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
from itertools import product

import torch
import random
import numpy as np
import os

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

LABEL_SET = [f"photo of a {c} with {d} artifacts, which is of {q} quality" for q, c, d in product(qualitys, scenes, dists_map)]
#LABEL_SET = [f"photo of a {c}, which is of {q} quality" for q, c in product(qualitys, scenes)]
print("Label set size: " + str(len(LABEL_SET)))
#LABEL_SET = ['bad photo', 'poor photo', 'fair photo', 'good photo', 'perfect photo']
DATASET_DIR = '/home/ccl/Datasets/koniq10k' #directory of dataset
IMG_DIR = os.path.join(DATASET_DIR, '1024x768')
SCORE_DIR = os.path.join(DATASET_DIR, 'koniq10k_scores_and_distributions.csv') #director of MOS score 
BATCH_SIZE = 3	 #select device for training and testing
DEVICE = "cuda:0" if is_available() else "cpu" #select device for training and testing
print(f"Using {DEVICE} device.")
#GRADIENT_CLIP = 10
EPOCHS = 100
CHECKPOINT_DIR = 'ckpt/mapleiqa_2ctx_liqe'
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best.pth')
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'latest.pth')

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def checkpoint(model, optimizer, ckpt_path, epoch=None, val_loss=None, load_ckpt=True):
    if load_ckpt:
        checkpoint = load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']

    else:
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, ckpt_path)
 
def train_step(dataloader, model, loss_fn, optimizer, scheduler=None):
    size = len(dataloader.dataset)
    model.train()
    for batch, (img, y) in enumerate(dataloader):

        # Compute prediction error
        pred = model(img)[0].squeeze()
        #print(pred)
        loss = loss_fn(pred, y)
            
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        #print([p.grad for p in model.parameters()])

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(y)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_step(dataloader, model, loss_fn, epoch, optimizer, best_loss=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, srocc_score, plcc_score = 0, 0, 0
    nan_srocc, nan_plcc = 0, 0

    with no_grad():
        for img, y in tqdm(dataloader):
            pred = model(img)[0].squeeze()
            test_loss += loss_fn(pred, y).item()

            temp_srocc = srocc(pred.cpu().numpy(), y.cpu().numpy())
            if np.isnan(temp_srocc):
                nan_srocc += 1
            else:
                srocc_score += temp_srocc

            temp_plcc = plcc(pred.cpu().numpy(), y.cpu().numpy())
            if np.isnan(temp_plcc):
                nan_plcc += 1
            else:
                plcc_score += temp_plcc

    test_loss /= num_batches

    if (num_batches == nan_srocc):
        print("All SROCC is undefined")
    else:
        srocc_score /= (num_batches-nan_srocc)

    if (num_batches == nan_plcc):
        print("All PLCC is undefined")
    else:
        plcc_score /= (num_batches-nan_plcc)

    print(f"Test Error:\n SROCC: {(srocc_score):>3f} with {nan_srocc} NaN samples, PLCC: {(plcc_score):>3f} with {nan_plcc} NaN samples, Val loss: {test_loss:>3f} \n")

    if test_loss < best_loss:
        best_loss = test_loss
        checkpoint(model,
                    optimizer,
                    BEST_CHECKPOINT_PATH,
                    epoch=epoch,
                    val_loss=test_loss,
                    load_ckpt=False)
        print(f"Saving best result to: {BEST_CHECKPOINT_PATH}")

    checkpoint(model,
                optimizer,
                LATEST_CHECKPOINT_PATH,
                epoch=epoch,
                val_loss=test_loss,
                load_ckpt=False)
    print(f"Saving latest result to: {LATEST_CHECKPOINT_PATH}")
    return best_loss
	
def preprocess(df):

    img_data = IMG_DIR + '/' + df['image_name'].to_numpy()
    mos_value = Tensor(df['MOS'].to_numpy())

    dataset = CustomKonIQ10kDataset(img_data, mos_value, device=DEVICE)
    return dataset

def run():

    score_df = read_csv(SCORE_DIR)
    dataset = preprocess(score_df)

    train_num = int(len(dataset)*0.8)
    test_num = len(dataset) - train_num
    train_set, val_set = random_split(dataset, [train_num, test_num])
    val_set.train_set = False

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=True)

    model = build_mapleiqa_v5(LABEL_SET)
    model.to(DEVICE)

    loss_fn = MSELoss()
    #optimizer = Adam(model.parameters())
    optimizer = AdamW(
        model.parameters(), lr=5e-6,
        weight_decay=0.001)

    scheduler = CosineAnnealingLR(optimizer, T_max=5)

    epoch, test_loss = 1, inf

    if os.path.exists(BEST_CHECKPOINT_PATH):
        epoch, test_loss = checkpoint(model,
                                    optimizer,
                                    BEST_CHECKPOINT_PATH)
        print(f"Loaded from {epoch} epoch, val loss: {test_loss:>3f}.")

    elif os.path.exists(LATEST_CHECKPOINT_PATH):
        epoch, test_loss = checkpoint(model,
                                    optimizer,
                                    LATEST_CHECKPOINT_PATH)
        print(f"Loaded from {epoch} epoch, val loss: {test_loss:>3f}.")

    else: print("Training new model.")

    for i in range(epoch, EPOCHS+1):
        print(f"Current epoch: {i}.")
        train_step(train_loader, model, loss_fn, optimizer, scheduler)
        test_loss = test_step(test_loader, model, loss_fn, (i+1), optimizer, best_loss=test_loss)

if __name__ == "__main__":
    run()