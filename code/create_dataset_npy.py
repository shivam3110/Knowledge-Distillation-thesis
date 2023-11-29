"""Extract MICCAI_BraTS2020 dataset as npz file.

python code/create_dataset_npy.py \
 --dataset_name "MICCAI_BraTS2020_Data" \
 --save_dir "MICCAI_BraTS20_trainigdata_npz_v2"\
 --save_as_format "png"
"""

import sys
import argparse
import glob
import os
import random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import imageio 
from tensorflow.keras.utils import to_categorical
import nibabel as nib
from sklearn.preprocessing import LabelEncoder 

from paths import DATASETS_DIR, MICCAI_BraTS2020_Data


print("The current DIR: ", os.getcwd())
np.random.seed(1000)
torch.manual_seed(1000)

def parse_args():
    """Parse th earguments and return args"""
    parser = argparse.ArgumentParser(description="Create dataset script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        required=True,
        help="Path to saving dir",
    )
    parser.add_argument(
        "--save_as_format",
        type=str,
        default=None,
        required=False,
        help="Format(png,jpg,npy)",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    #Sanity Checks
    if args.dataset_name is None:
        raise ValueError("Need a Dataset to load.")
    if args.save_dir is None:
        raise ValueError("Need a Saving Folder.")
    return args


def save_nii_mask_to_npy(mask_file,subject_save_path, save_as):
    if not os.path.exists(subject_save_path):
        os.makedirs(subject_save_path)       
    subject_slice_id = subject_save_path.name

    mask_file[mask_file==4] = 3 
    mask = np.moveaxis(mask_file, 2,0)

    labelencoder = LabelEncoder()
    n,h,w = mask.shape
    n_classes = 4
    mask_reshape = mask.reshape(-1,1)
    mask_encoded = labelencoder.fit_transform(mask_reshape)
    mask_encoded_shape = mask_encoded.reshape(n,h,w)
    for i in range(n):
        silce = mask_encoded_shape[i,:,:]
        mask_img = np.expand_dims(silce , axis = -1)
        if not save_as:
            np.save(os.path.join(str(subject_save_path),str(subject_slice_id)+'_{}.npy'.format(i)), mask_img)
        else:
            imageio.imwrite(os.path.join(str(subject_save_path),str(subject_slice_id)+'_{}.png'.format(i)), mask_img)
    return

def save_nii_to_npy(subject_data, subject_save_path, save_as):           
    if not os.path.exists(subject_save_path):        
        os.makedirs(subject_save_path)        
    subject_slice_id = subject_save_path.name   
    (x,y,z) = subject_data.shape
    for i in range(z):      #z Is a sequence of images 
        silce = subject_data[:, :, i]   # You can choose which direction of slice 
        np.expand_dims(silce, axis =-1)
        if not save_as:
            np.save(os.path.join(str(subject_save_path),str(subject_slice_id)+'_{}.npy'.format(i)), silce)
        else:
            imageio.imwrite(os.path.join(str(subject_save_path),str(subject_slice_id)+'_{}.png'.format(i)), silce)
    return

def extract_2D_slice(data_dir, save_dir, save_as:str):
    subject_ids = sorted(os.listdir(data_dir))
    substring_subject_id ='BraTS20_'
    csv_li = []
    if subject_ids is not None:
        for subject in subject_ids:
            if substring_subject_id in subject:                        
                subject_path = data_dir / subject
                subject_modularities = sorted(os.listdir(subject_path))
                for modularity in subject_modularities:                                               
                    subject_data = nib.load(os.path.join(subject_path,modularity)).get_fdata()
                    mod_name = modularity.replace('.nii.gz','')
                    subject_save_path = save_dir / subject / mod_name
                    if 'seg' in mod_name:
                        save_nii_mask_to_npy(subject_data, subject_save_path, save_as)
                    else:
                        save_nii_mask_to_npy(subject_data,subject_save_path, save_as)
            elif subject.endswith('.csv'):            
                csv_li.append(subject)
                # print("No data-file found. Only found",len(csv_li), "files.")
    else:
        print("Empty DIR. No '.csv/data-file' present.")
    return

def main():
    
    args = parse_args()
    DATASET_NAME = args.dataset_name
    output_dir = Path(args.save_dir)

    data_dir = DATASETS_DIR / DATASET_NAME
    saving_dir = DATASETS_DIR /output_dir
    save_as = args.save_as_format
    extract_2D_slice(data_dir, saving_dir, save_as)
 
if __name__ == "__main__":
    main()  

    # path = "knowledge_distillation/dataset/scratch/MICCAI_BraTS2020_TrainingData_2D/"

    # naming_csv_path ='knowledge_distillation/dataset/scratch/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv'
    # df_name =  pd.read_csv(naming_csv_path)
    # df_name =  df_name.drop(['BraTS_2017_subject_ID','BraTS_2018_subject_ID','TCGA_TCIA_subject_ID','BraTS_2019_subject_ID'], axis =1)

 