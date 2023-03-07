import faiss
import os
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torchvision import models

from torchvision.transforms import Compose, transforms
from PIL import Image
import cv2
import sqlite3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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


def model_output(image_path):
    img = cv2.imread(image_path)
    imgt = transform(img)
    imgt = imgt.to(device)
    with torch.no_grad():
        inference = as_numpy(model(torch.unsqueeze(imgt[0], 0)))
    return inference


INDEX_PATH =  "/nethome/kravicha3/aryan/project/notebooks/redditdataset/saves/HNSW_dataindex.index"
index = faiss.read_index(INDEX_PATH)
k = 10
REDDIT_DATA_HOME = os.path.join("/nethome/kravicha3/aryan/project/dataset/Reddit_Provenance_Datasets/data/")
reddit_threads = os.listdir(REDDIT_DATA_HOME)
# reddit_threads = ['_This_picture_of_Hillary_Clinton_and_Barack_Obama']
c = sqlite3.connect("/nethome/kravicha3/aryan/project/notebooks/redditdataset/saves/eva_catalog.db").cursor()

def get_similarity_results(image_path):
    # get model output
    output = model_output(image_path)
    # get index similarity
    D, I = index.search(output, k)
    # get the filenames
    results = list()
    for i in I[0]:
        c.execute(f"SELECT * FROM '192111ccbbbfc5042415841dfaa9f90a' WHERE _row_id={i}")
        r= c.fetchall()
        results.append(r[0][1])
    return results


def get_examples():
    dataframe = pd.DataFrame(columns=['image_name', 'image_dir', 'num_similar', 'results'])
    # take a directory and get all image results from it
    for dir_name in reddit_threads:
        img_dir = os.path.join(REDDIT_DATA_HOME, dir_name)
        for file_name in os.listdir(img_dir):
            # store and save results some where
            if (file_name.endswith(".jpg") or file_name.endswith(".png")):
                fp = os.path.join(img_dir, file_name)
                try:
                    result = get_similarity_results(fp)
                except Exception as e:
                    print(e)
                    continue
                num_sim = 0
                
                for name in result:
                    
                    if img_dir in name:
                        num_sim += 1
                
                dataframe.loc[len(dataframe)] = [file_name, img_dir, num_sim, result]
                torch.cuda.empty_cache()
            
        print(f"completed dir: {dir_name}", len(dataframe))

    return dataframe

output_frame = get_examples()
print(len(output_frame))
output_frame.to_pickle("hnsw_all_image_results.pkl")