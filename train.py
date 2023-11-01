from models.maple_iqa import build_mapleiqa
from torch_utils.dataset import CustomKonIQ10kDataset, CustomTIDDataset, CustomCSIQDataset, CustomSPAQDataset, CustomLIVEDataset, CustomKADID10kDataset
from torch_utils.metrics import srocc, plcc
import configparser
from ast import literal_eval
import scipy.io

from torch.utils.data import DataLoader, random_split
from pandas import read_csv, read_excel, DataFrame
from numpy import where, unique, apply_along_axis, inf
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

config = configparser.ConfigParser()
config.read('/home/ccl/Code/MaPLe-IQA/CLIP/config_gen/mapleiqa_sample_2.ini')

LABEL_SET = literal_eval(config['TRAINING_CONFIG']['LABEL_SET'])
print("Label set size: " + str(len(LABEL_SET)))
KONIQ10K_IMG_DIR = config['TRAINING_CONFIG']['KONIQ10K_IMG_DIR']
KONIQ10K_SCORE_DIR = config['TRAINING_CONFIG']['KONIQ10K_SCORE_DIR'] #director of MOS score

TID2008_IMG_DIR = config['TRAINING_CONFIG']['TID2008_IMG_DIR']
TID2008_SCORE_DIR = config['TRAINING_CONFIG']['TID2008_SCORE_DIR']

CSIQ_IMG_DIR = config['TRAINING_CONFIG']['CSIQ_IMG_DIR']
CSIQ_SCORE_DIR = config['TRAINING_CONFIG']['CSIQ_SCORE_DIR']

TID2013_IMG_DIR = config['TRAINING_CONFIG']['TID2013_IMG_DIR']
TID2013_SCORE_DIR = config['TRAINING_CONFIG']['TID2013_SCORE_DIR']

SPAQ_IMG_DIR = config['TRAINING_CONFIG']['SPAQ_IMG_DIR']
SPAQ_SCORE_DIR = config['TRAINING_CONFIG']['SPAQ_SCORE_DIR']

LIVE_IMG_MAT = config['TRAINING_CONFIG']['LIVE_IMG_MAT'] 
LIVE_SCORE_MAT = config['TRAINING_CONFIG']['LIVE_SCORE_MAT']
LIVE_IMG_DIR = config['TRAINING_CONFIG']['LIVE_IMG_DIR']

KADID10K_IMG_DIR = config['TRAINING_CONFIG']['KADID10K_IMG_DIR']
KADID10K_SCORE_DIR = config['TRAINING_CONFIG']['KADID10K_SCORE_DIR']

BATCH_SIZE = int(config['TRAINING_CONFIG']['BATCH_SIZE'])	 #select device for training and testing
DEVICE = config['TRAINING_CONFIG']['DEVICE'] #select device for training and testing
print(f"Using {DEVICE} device.")
#GRADIENT_CLIP = 10
EPOCHS = int(config['TRAINING_CONFIG']['EPOCHS'])
CHECKPOINT_DIR = config['TRAINING_CONFIG']['CHECKPOINT_DIR']
BEST_CHECKPOINT_PATH = config['TRAINING_CONFIG']['BEST_CHECKPOINT_PATH']
LATEST_CHECKPOINT_PATH = config['TRAINING_CONFIG']['LATEST_CHECKPOINT_PATH']

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def checkpoint(model, optimizer, scheduler, ckpt_path, epoch=None, val_loss=None, load_ckpt=True):
    if load_ckpt:
        checkpoint = load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']

    else:
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        if scheduler: state_dict['scheduler_state_dict'] = scheduler.state_dict()
        save(state_dict, ckpt_path)
 
def train_step(dataloader, model, loss_fn, optimizer, scheduler=None):
    model.train()
    for key, each_dataloader in dataloader.items():
        if '.train' in key:
            pbar = tqdm(each_dataloader)
            for img, y in pbar:

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
                pbar.set_description(f"Current loader: {key}, Current loss: {loss.item()}")

def test_step(dataloader, model, loss_fn, epoch, optimizer, best_loss=None, scheduler=None):
    num_batches = len(dataloader)
    model.eval()
    loss_list = []

    with no_grad():
        for key, each_dataloader in dataloader.items():
            if '.test' in key:
                test_loss = 0
                pred_list, true_list = [], []
                for img, y in tqdm(each_dataloader):
                    pred = model(img)[0].squeeze()
                    test_loss += loss_fn(pred, y).item()

                    pred_list.append(pred.cpu().numpy())
                    true_list.append(y.cpu().numpy())

                srocc_score = srocc(np.concatenate(pred_list), np.concatenate(true_list))
                plcc_score = plcc(np.concatenate(pred_list), np.concatenate(true_list))
                
                test_loss /= num_batches
                print(f"Dataset: {key} MSE: {(test_loss):>3f} SROCC: {(srocc_score):>3f}, PLCC: {(plcc_score):>3f}\n")
                loss_list.append(test_loss)

    test_loss = sum(loss_list)/len(loss_list)
    if test_loss < best_loss:
        best_loss = test_loss
        checkpoint(model,
                    optimizer,
                    scheduler,
                    BEST_CHECKPOINT_PATH,
                    epoch=epoch,
                    val_loss=test_loss,
                    load_ckpt=False)
        print(f"Saving best result to: {BEST_CHECKPOINT_PATH}")

    checkpoint(model,
                optimizer,
                scheduler,
                LATEST_CHECKPOINT_PATH,
                epoch=epoch,
                val_loss=test_loss,
                load_ckpt=False)
    print(f"Saving latest result to: {LATEST_CHECKPOINT_PATH}")
    return best_loss
	
def preprocess(koniq10k_df, tid2008_df, tid2013_df, spaq_df, live_df, kadid10k_df):

    koniq10k_img = KONIQ10K_IMG_DIR + '/' + koniq10k_df['image_name'].to_numpy()
    koniq10k_mos = Tensor(koniq10k_df['MOS'].to_numpy())

    tid2008_mos = tid2008_df[0].to_numpy()
    tid2008_img = (TID2008_IMG_DIR + '/' + tid2008_df[1]).to_numpy()
    
    #csiq_mos = csiq_df['dmos'].to_numpy()
    #csiq_img = (CSIQ_IMG_DIR + '/' + csiq_df['dst_type'].str.replace(' ', '') + "/" + csiq_df['image'] \
    #+ "." + csiq_df['dst_type'].apply(lambda x: 'AWGN' if x == "noise" else x).apply(lambda x: x.upper() if x in ["jpeg", "blur"] else x).str.replace(' ', '')\
     #+ "." + csiq_df['dst_lev'].apply(str) + ".png").to_numpy()
    
    tid2013_mos = tid2013_df[0].to_numpy()
    tid2013_img = (TID2013_IMG_DIR + '/' + tid2013_df[1]).to_numpy()

    spaq_img = SPAQ_IMG_DIR + '/' + spaq_df['Image name'].to_numpy()
    spaq_mos = Tensor(spaq_df['MOS'].to_numpy())

    live_img = LIVE_IMG_DIR + '/' + (live_df['name'].apply(str).str.replace("'", "")).str.strip('[]').to_numpy()
    live_mos = Tensor(live_df['mos'].to_numpy())

    kadid10k_img = KADID10K_IMG_DIR + '/' + kadid10k_df['dist_img'].to_numpy()
    kadid10k_mos = Tensor(kadid10k_df['dmos'].to_numpy())


    koniq10k_dataset = CustomKonIQ10kDataset(koniq10k_img, koniq10k_mos, device=DEVICE)
    tid2008_dataset = CustomTIDDataset(tid2008_img, tid2008_mos, device=DEVICE)
    #csiq_dataset = CustomCSIQDataset(csiq_img, csiq_mos, device=DEVICE)
    tid2013_dataset = CustomTIDDataset(tid2013_img, tid2013_mos, device=DEVICE)
    spaq_dataset = CustomSPAQDataset(spaq_img, spaq_mos, device=DEVICE)
    live_dataset = CustomLIVEDataset(live_img, live_mos, device=DEVICE)
    kadid10k_dataset = CustomKADID10kDataset(kadid10k_img, kadid10k_mos, device=DEVICE)

    return {'koniq10k' : koniq10k_dataset, 
            'tid2008' : tid2008_dataset,
            #'csiq' : csiq_dataset,
            'spaq': spaq_dataset,
            'tid2013' : tid2013_dataset,
            'live': live_dataset,
            'kadid10k': kadid10k_dataset}

def run():

    koniq10k_df = read_csv(KONIQ10K_SCORE_DIR)
    tid2008_df = read_csv(TID2008_SCORE_DIR, sep=" ", header=None)
    #csiq_df = read_csv(CSIQ_SCORE_DIR, sep="\t")
    tid2013_df = read_csv(TID2013_SCORE_DIR, sep=" ", header=None)
    spaq_df = read_excel(SPAQ_SCORE_DIR)
    kadid10k_df = read_csv(KADID10K_SCORE_DIR)

    live_mos_mat = scipy.io.loadmat(LIVE_SCORE_MAT)
    live_img_mat = scipy.io.loadmat(LIVE_IMG_MAT)

    live_df = DataFrame(data = live_img_mat['AllImages_release'], columns = ['name'])
    live_df['mos'] = live_mos_mat['AllMOS_release'][0]
    
    datasets = preprocess(koniq10k_df, tid2008_df, tid2013_df, spaq_df, live_df, kadid10k_df)
    dataloaders = {}

    for key, dataset in datasets.items():
        train_num = int(len(dataset)*0.8)
        test_num = len(dataset) - train_num 
        train_set, val_set = random_split(dataset, [train_num, test_num])
        val_set.train_set = False

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

        dataloaders[key+'.train'] = train_loader
        dataloaders[key+'.test'] = test_loader

    model = build_mapleiqa(LABEL_SET, config['MODEL_CONFIG']['MODEL'])
    model.to(DEVICE)

    loss_fn = MSELoss()
    #optimizer = Adam(model.parameters())
    optimizer = AdamW(
        model.parameters(), lr=5e-6,
        weight_decay=0.001)

    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    #scheduler = None

    epoch, test_loss = 1, inf

    if os.path.exists(BEST_CHECKPOINT_PATH):
        epoch, test_loss = checkpoint(model,
                                    optimizer,
                                    scheduler,
                                    BEST_CHECKPOINT_PATH)
        print(f"Loaded from {epoch} epoch, avg val loss: {test_loss:>3f}.")

    elif os.path.exists(LATEST_CHECKPOINT_PATH):
        epoch, test_loss = checkpoint(model,
                                    optimizer,
                                    scheduler,
                                    LATEST_CHECKPOINT_PATH)
        print(f"Loaded from {epoch} epoch, avg val loss: {test_loss:>3f}.")

    else: print("Training new model.")

    for i in range(epoch, EPOCHS+1):
        print(f"Current epoch: {i}.")
        train_step(dataloaders, model, loss_fn, optimizer, scheduler)
        test_loss = test_step(dataloaders, model, loss_fn, (i+1), optimizer, best_loss=test_loss, scheduler=scheduler)

if __name__ == "__main__":
    run()