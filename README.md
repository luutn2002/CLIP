# MaPLe-IQA - A deep learning model for blind image quality assessment.

[Paper](https://ieeexplore.ieee.org/document/10402183)

A deep learning models for blind image quality assessment based on [MaPLe(Multimodal Promt Learning)](https://arxiv.org/pdf/2210.03117.pdf), a CLIP based
model with added coupling function for performance enhancement.

## Quickstart

This is a quickstart guide to use MaPLe-IQA for quick deployment

### Step 1: Install the repository as a package

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm scipy pandas
$ pip install git+https://github.com/luutn2002/mapleiqa.git
```

### Step 2: Create a .ini file for model config

```python
import configparser
import os
from torch.cuda import is_available

#Script used to create configuration for models.

config = configparser.ConfigParser()
TEST_MODEL_NAME = 'mapleiqa'
DEVICE = "cuda:0" if is_available() else "cpu"

config['MODEL_CONFIG'] = {
  'MAPLE_PROMPT_DEPTH' : 9,
  'MAPLE_INPUT_SIZE' : (224, 224),
  'MAPLE_N_CTX' : 2,
  'MAPLE_CTX_INIT' : "This is a ",
  'MAPLE_PRETRAIN_DIR' : './model.pth.tar-5',
  'MAPLE_POS_EMBED' : True,
  'MAPLE_INNER_BATCH' : 12,
  'BACKBONE' : 'ViT-B/32',
  'DEVICE' : DEVICE,
  'MODEL': 'MaPLeIQA',
  'FREEZE_IMAGE_ENCODER': False,
  'FREEZE_TEXT_ENCODER': False
}

with open(f"./configs/{config['DEFAULT']['TEST_MODEL_NAME']}.ini", 'w') as configfile:
  config.write(configfile)
```
Your .ini file should look like this:

```dosini
[MODEL_CONFIG]
maple_prompt_depth = 9
maple_input_size = (224, 224)
maple_n_ctx = 2
maple_ctx_init = This is a 
maple_pos_embed = True
maple_inner_batch = 12
backbone = ViT-B/32
device = cuda:0
model = MaPLeIQA
freeze_image_encoder = False
freeze_text_encoder = False
```

### Step 3: Include model in your code

```python
from mapleiqa.models import build_mapleiqa
import torch

model = build_mapleiqa(["good photo", "bad photo"], #List of prompts equivalent to number of MaPLe models being used
                       "MaPLeIQA", 
                       "./configs/mapleiqa.ini")

ckpt = torch.load("./checkpoint_sample_2.pt") #Pretrained checkpoint
model.load_state_dict(ckpt['model_state_dict']) #Load our pretrained models if you prefer
```
Check our [example scripts](https://github.com/luutn2002/mapleiqa/blob/master/examples/scripts/train_example.py) for more detailson how we train the model.

## Result with datasets

| Datasets  | SROCC | PLCC  |
|-----------|:-----:|:-----:|
| KonIQ-10K | 0.648 | 0.701 |
| TID 2008  | 0.934 | 0.929 |
| SPAQ      | 0.796 | 0.801 |
| TID 2013  | 0.912 | 0.914 |
| LIVE      | 0.560 | 0.583 |
| KADID-10K | 0.930 | 0.926 |

> Notes: Datasets are train all at once in sequentials.

## Citation

```bibtex
@INPROCEEDINGS{10402183,
  author={Luu, Nhan T. and Onuoha, Chibuike and Thang, Truong Cong},
  booktitle={2023 IEEE 15th International Conference on Computational Intelligence and Communication Networks (CICN)}, 
  title={Blind Image Quality Assessment With Multimodal Prompt Learning}, 
  year={2023},
  pages={614-618},
  doi={10.1109/CICN59264.2023.10402183}}
```