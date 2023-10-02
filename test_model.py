import torch
from models.maple_iqa import build_mapleiqa

LABEL_SET = ['good photo', 'bad photo']
DEVICE = "cpu"

checkpoint = torch.load('./model.pth.tar-5', map_location=DEVICE)
print("Model's state_dict:")
for param_tensor in checkpoint['state_dict']:
    print(param_tensor, "\t", checkpoint['state_dict'][param_tensor].size())