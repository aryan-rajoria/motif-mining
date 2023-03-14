import faiss
import os
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torchvision import models

from torchvision.transforms import Compose, transforms
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image
import cv2
import sqlite3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

DATASET = "/nethome/kravicha3/aryan/project/dataset/Reddit_Provenance_Datasets/data/"
# DATASET = "/nethome/kravicha3/aryan/project/dataset/IndonesianMeme/final/TwitterImages"


# model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights)
model = models.resnet50(pretrained=True, progress=False)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Identity()
model.to(device)
model.eval()

def transform(images: np.ndarray):
    transformed = [transforms.ToTensor()]
    composed = Compose(transformed)
    return composed(Image.fromarray(images[:, :, ::-1])).unsqueeze(0)

def as_numpy(val: Tensor) -> np.ndarray:
    return val.detach().cpu().numpy()

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
        inference = as_numpy(model(torch.unsqueeze(imgt[0], 0)))
    return inference

def get_image_features(start_path = '.'):
    df = pd.DataFrame(columns=['img_name', 'dir', 'features'])
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
        
            if (fp.endswith('.jpg') or fp.endswith('.png') or fp.endswith('.mp4')):
                try:
                    features = model_output(fp)
                except Exception as e:
                    print(e, fp)
                    continue
                df.loc[len(df.index)] = [f, dirpath, features]
            
            torch.cuda.empty_cache()
    return df


feat_df = get_image_features(DATASET)
feat_df.to_pickle('./tune_result/reddit_resnet_df.pkl')

# DATASET = "/nethome/kravicha3/aryan/project/dataset/IndonesianMeme/final/TwitterImages"

# feat_df = get_image_features(DATASET)
# feat_df.to_pickle('./tune_result/reddit_resnet_df.pkl')
