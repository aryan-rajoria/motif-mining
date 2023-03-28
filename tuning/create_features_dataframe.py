import faiss
import os
import numpy as np
import pandas as pd
from typing import Dict

import torch
from torch import Tensor
from torchvision import models
from torch import nn

from torchvision.transforms import Compose, transforms
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image
import cv2
import sqlite3

from torchvision.models.feature_extraction import create_feature_extractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

DATASET = "/nethome/kravicha3/aryan/project/dataset/Reddit_Provenance_Datasets/data/"
# DATASET = "/nethome/kravicha3/aryan/project/dataset/IndonesianMeme/final/TwitterImages"


# model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights)
model_resnet = models.resnet50(pretrained=True, progress=False)
return_nodes_resnet = ['layer1.2', 'layer2.3', 'layer3.5', 'layer4.2']

layer1_pool = nn.AdaptiveAvgPool2d((4, 4))
layer2_pool = nn.AdaptiveAvgPool2d((5, 7))
layer3_pool = nn.AdaptiveAvgPool2d((5, 7))
layer4_pool = nn.AdaptiveAvgPool2d((5, 7))

feature_extractor_resnet = create_feature_extractor(model_resnet, return_nodes=return_nodes_resnet)
for param in model_resnet.parameters():
    param.requires_grad = False
model_resnet.fc = torch.nn.Identity()
model_resnet.to(device)
model_resnet.eval()

def transform(images: np.ndarray):
    transformed = [transforms.ToTensor()]
    composed = Compose(transformed)
    return composed(Image.fromarray(images[:, :, ::-1])).unsqueeze(0)

def as_numpy(val: Tensor) -> np.ndarray:
    return val.detach().cpu().numpy()

def dict_as_numpy(inference: Dict) -> Dict:
    for layer_output in inference:
        output = inference[layer_output]
        inference[layer_output] = as_numpy(output)
    return inference

def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        cap = cv2.VideoCapture(image_path)
        ret, img = cap.read()
        cap.release()
    return img

def model_output(image_path):
    img = read_image(image_path)
    imgt = transform(img)
    imgt = imgt.to(device)
    with torch.no_grad():
        inference = as_numpy(model_resnet(torch.unsqueeze(imgt[0], 0)))
    return inference

def feature_extractor_output(image_path):
    img = read_image(image_path)
    imgt = transform(img)
    imgt = imgt.to(device)
    with torch.no_grad():
        inferences = dict_as_numpy(feature_extractor_resnet(torch.unsqueeze(imgt[0], 0)))
    return inferences

def get_image_features(start_path = '.'):
    df = pd.DataFrame(columns=['img_name', 'dir', 'features1', 'features2', 'features3', 'features4'])
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
        
            if (fp.endswith('.jpg') or fp.endswith('.png') or fp.endswith('.mp4')):
                try:
                    # features = model_output(fp)
                    features = feature_extractor_output(fp)
                    
                    features['layer1.2'] = 
                except Exception as e:
                    print(e, fp)
                    continue
                df.loc[len(df.index)] = [f, dirpath, features['layer1.2'], features['layer2.3'], features['layer3.5'], features['layer4.2']]
            
            torch.cuda.empty_cache()
            break
        # break
    return df


feat_df = get_image_features(DATASET)
feat_df.to_pickle('./tune_result/multi_reddit_resnet_df.pkl')

# DATASET = "/nethome/kravicha3/aryan/project/dataset/IndonesianMeme/final/TwitterImages"

# feat_df = get_image_features(DATASET)
# feat_df.to_pickle('./tune_result/reddit_resnet_df.pkl')
