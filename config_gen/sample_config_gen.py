import configparser
import os
from torch.cuda import is_available

config = configparser.ConfigParser()
TEST_MODEL_NAME = 'mapleiqa_sample_2'
DEVICE = "cuda:0" if is_available() else "cpu"
#DEVICE = "cpu"

config['DEFAULT']= {
  'TEST_MODEL_NAME' : TEST_MODEL_NAME,
  'KONIQ10K_DIR' : '/home/ccl/Datasets/koniq10k', #directory of dataset
  'TID2008_DIR' : '/home/ccl/Datasets/tid2008',
  'CSIQ_DIR' : '/home/ccl/Datasets/csiq',
  'TID2013_DIR' : '/home/ccl/Datasets/tid2013',
  'SPAQ_DIR' : '/home/ccl/Datasets/spaq',
  'LIVE_DIR': '/home/ccl/Datasets/live1/ChallengeDB_release',
  'KADID10K_DIR' : '/home/ccl/Datasets/kadid10k',
  'CHECKPOINT_DIR' : f'ckpt/{TEST_MODEL_NAME}',
}
config['TRAINING_CONFIG']= {
  'LABEL_SET' : ["bad photo", "good photo"],

  'KONIQ10K_IMG_DIR' : os.path.join(config['DEFAULT']['KONIQ10K_DIR'], '1024x768'),
  'KONIQ10K_SCORE_DIR' : os.path.join(config['DEFAULT']['KONIQ10K_DIR'], 'koniq10k_scores_and_distributions.csv'), #director of MOS score

  'TID2008_IMG_DIR' : os.path.join(config['DEFAULT']['TID2008_DIR'], 'distorted_images'),
  'TID2008_SCORE_DIR' : os.path.join(config['DEFAULT']['TID2008_DIR'], 'mos_with_names.txt'),

  'CSIQ_IMG_DIR' : os.path.join(config['DEFAULT']['CSIQ_DIR'], 'img'),
  'CSIQ_SCORE_DIR' : os.path.join(config['DEFAULT']['CSIQ_DIR'], 'mos.csv'),

  'TID2013_IMG_DIR' : os.path.join(config['DEFAULT']['TID2013_DIR'], 'distorted_images'),
  'TID2013_SCORE_DIR' : os.path.join(config['DEFAULT']['TID2013_DIR'], 'mos_with_names.txt'),

  'SPAQ_IMG_DIR' : os.path.join(config['DEFAULT']['SPAQ_DIR'], 'TestImage'),
  'SPAQ_SCORE_DIR' : os.path.join(config['DEFAULT']['SPAQ_DIR'], 'mos.xlsx'),

  'LIVE_IMG_MAT' : os.path.join(config['DEFAULT']['LIVE_DIR'], 'Data/AllImages_release.mat'),
  'LIVE_SCORE_MAT' : os.path.join(config['DEFAULT']['LIVE_DIR'], 'Data/AllMOS_release.mat'),
  'LIVE_IMG_DIR' : os.path.join(config['DEFAULT']['LIVE_DIR'], 'Images'),

  'KADID10K_IMG_DIR' : os.path.join(config['DEFAULT']['KADID10K_DIR'], 'images'),
  'KADID10K_SCORE_DIR' : os.path.join(config['DEFAULT']['KADID10K_DIR'], 'dmos.csv'),

  'BATCH_SIZE' : 32,
  'GRADIENT_CLIP' : 10,
  'EPOCHS' : 10,
  'BEST_CHECKPOINT_PATH' : os.path.join(config['DEFAULT']['CHECKPOINT_DIR'], 'best.pth'),
  'LATEST_CHECKPOINT_PATH' : os.path.join(config['DEFAULT']['CHECKPOINT_DIR'], 'latest.pth'),
  'DEVICE' : DEVICE
}

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
  'MODEL': 'MaPLeIQAPredictor_V3',
  'FREEZE_IMAGE_ENCODER': True,
  'FREEZE_TEXT_ENCODER': True
}

with open(f"{config['DEFAULT']['TEST_MODEL_NAME']}.ini", 'w') as configfile:
  config.write(configfile)