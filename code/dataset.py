import time
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision.transforms import Resize

from paths import DATAFRAME_DIR

np.random.seed(31101995)
torch.manual_seed(31101995)

def class_to_idx(label):
    idx = 0
    if label == 'healthy':idx =0
    if label == 'HGG':idx =1
    if label=='LGG': idx = 2        
    return idx

class CustomDataset_from_csv(Dataset):
    def __init__(self, data_df , mod, transform = None , label_transform= None):
        self.dataframe = data_df #pd.read_csv(annotations_file)
        self.mod = mod
        self.transform = transform
        self.label_transform = label_transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self , idx):
        img_path =  self.dataframe.loc[idx,'image_path'] 
        img_path_mod = img_path. replace('seg',f'{self.mod}')
        # print(img_path_mod)
        img = np.load(img_path_mod)
        grayscale_image = np.resize(img, (224,224))
        image = np.repeat(grayscale_image[..., np.newaxis], 3, -1)
        # print(np.shape(image))
        # Convert numpy array to torch.Tensor and set the data type to torch.float32
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # print(np.shape(image_tensor))
        # image = image.astype('float32')

        label_col = 'label_' + str(self.mod)     
        label_mod = self.dataframe.loc[idx, label_col]
        class_label = class_to_idx(label_mod)
        label = torch.tensor([class_label])
                             
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.label_transform:
            label = self.target_transform(label)
        return(image_tensor, label )


class CustomDataset_multimodal_from_csv(Dataset):
    def __init__(self, data_df , mod, transform = None , label_transform= None):
        self.dataframe = data_df #pd.read_csv(annotations_file)
        self.mod = mod
        self.transform = transform
        self.label_transform = label_transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self , idx):
        img_path =  self.dataframe.loc[idx,'image_path'] 
        img_path_flair = img_path.replace('seg','flair')
        img_path_t1ce = img_path.replace('seg','t1ce')
        img_path_t2 = img_path.replace('seg','t2')
        img_flair = np.load(img_path_flair)
        img_t1ce = np.load(img_path_t1ce)
        img_t2 = np.load(img_path_t2)

        img_flair_resize = np.repeat(np.resize(img_flair, (224,224))[..., np.newaxis], 1, -1 )
        img_t1ce_resize =np.repeat(np.resize(img_t1ce, (224,224))[..., np.newaxis], 1, -1 )
        img_t2_resize = np.repeat(np.resize(img_t2, (224,224))[..., np.newaxis], 1, -1 )

        img_multimodal =  np.concatenate((img_flair_resize, img_t1ce_resize, img_t2_resize), axis=-1)
        image_tensor = torch.from_numpy(img_multimodal).permute(2, 0, 1).float()

        # image = np.repeat(grayscale_image[..., np.newaxis], 3, -1)
        # img_multimodal = img_multimodal.astype('float32')
        label_col = 'label_' + str(self.mod)   
        label_mod = self.dataframe.loc[idx, label_col]
        class_label = class_to_idx(label_mod)
        label = torch.tensor([class_label])
                             
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.label_transform:
            label = self.target_transform(label)
        return(image_tensor, label )

def get_distribution_ratio(split_dict):
    """Calculate the distribution ratio of a given split.

    Parameters
    ----------
    split_dict : dict
        A dictionary containing the number of samples for each class.
    
    Returns
    -------
    dict
        A dictionary containing the ratio of samples for each class, as a float
        with 2 decimal places.
    """

    total = sum(split_dict.values())
    result = {key: '{:.2f}'.format(value / total) for key, value in split_dict.items()}
    print(result)
    return result
    

def get_class_distribution(train_df, test_df, val_df, mod, output_dir):
    """Create dataframe for class-distribution for each split(train, val, test).        
    """
    label_col = f'label_{mod}'
    train_dict = train_df[label_col].value_counts().to_dict()
    val_dict = val_df[label_col].value_counts().to_dict()
    test_dict = test_df[label_col].value_counts().to_dict()
    index_labels=['train','val','test']
    class_distribution_df = pd.DataFrame([train_dict, val_dict,test_dict], 
                                                   index=index_labels,
                                                  )
    class_distribution_ratio_df = pd.DataFrame([get_distribution_ratio(train_dict), 
                                                 get_distribution_ratio(val_dict),
                                                 get_distribution_ratio(test_dict)], 
                                                     index=index_labels,
                                                    )   
    class_distribution_df.to_csv(output_dir / 'class_distribution_df.csv', index =False)
    class_distribution_ratio_df.to_csv(output_dir / 'class_distribution_ratio_df.csv', index =False)
    return class_distribution_df, class_distribution_ratio_df

def get_split_ratios(subject_list, training_subjects, val_subjects, test_subject  ):
    train_ratio = len(training_subjects)/len(subject_list)
    test_ratio = len(test_subject)/len(subject_list)
    val_ratio = len(val_subjects)/len(subject_list)
    print("train_split :", '{:.3f}'.format(train_ratio),"test_split :",  '{:.3f}'.format(test_ratio), "val_split :", '{:.3f}'.format(val_ratio))
    print("train_samples :",len(training_subjects), "test_samples :",len(test_subject), "val_samples :",len(val_subjects))
    return train_ratio, val_ratio, test_ratio

def split_dataset(data_df, test_split, val_split, mod, output_dir):
    test_size = test_split + val_split
    val_size = 0.1 / test_size 
    np.random.seed(31101995)
    torch.manual_seed(31101995)
#     print("test_size:", test_size, "val_size:", val_size)
    #Using pateind_id to split the dataset into train, test and validation overfits the validation set.
    #Therefore TRY_2: using the entire data(shuffled) for splitting
    subject_list = np.unique(data_df.subject_id).tolist()
    
    # training_subjects, test_subjects = train_test_split(subject_list ,test_size = test_size, random_state = 42, shuffle = True)
    # test_subject, val_subjects = train_test_split(test_subjects ,test_size = val_size, random_state = 42, shuffle = True)

    # #Print split ratios and number of subjects per split

    # train_ratio, val_ratio, test_ratio = get_split_ratios(subject_list, training_subjects,val_subjects, test_subject  )
    # train_df = data_df[data_df['subject_id'].isin(training_subjects)].reset_index(drop=True)
    # test_df = data_df[data_df['subject_id'].isin(test_subject)].reset_index(drop=True)
    # val_df = data_df[data_df['subject_id'].isin(val_subjects)].reset_index(drop=True)

    training_df, test_df = train_test_split(data_df ,test_size = 0.12, random_state = 42, shuffle = True)
    train_df , val_df = train_test_split(training_df,test_size = 0.2, random_state = 42, shuffle = True)

    train_ratio, val_ratio, test_ratio = get_split_ratios(subject_list, training_df.subject_id, val_df.subject_id, test_df.subject_id  )
    train_df.reset_index(drop=True, inplace=True) 
    val_df.reset_index(drop=True, inplace=True) 
    test_df.reset_index(drop=True, inplace=True) 
    train_df.to_csv( DATAFRAME_DIR / 'train_df.csv' )
    val_df.to_csv(DATAFRAME_DIR / 'val_df.csv')
    test_df.to_csv(DATAFRAME_DIR / 'test_df.csv')

    get_class_distribution(train_df, test_df, val_df, mod, output_dir)
    return train_df, val_df, test_df

def get_splits(dataframe_dir =  DATAFRAME_DIR):
    """ Reads and returns train, validation, and test splits of a dataset.

    Parameters
    ----------
    dataframe_dir : str, optional
        Directory path containing the CSV files for train, validation, and test splits.
        Default is the value of the global constant DATAFRAME_DIR.

    Returns
    -------
    tuple
        A tuple containing three Pandas dataframes, in the following order:
        - train_df : dataframe
            The training split dataframe.
        - val_df : dataframe
            The validation split dataframe.
        - test_df : dataframe
            The test split dataframe.
    """
    train_df_path = dataframe_dir / 'train_df.csv'
    val_df_path = dataframe_dir / 'val_df.csv'
    test_df_path = dataframe_dir / 'test_df.csv'
    try:
        train_df = pd.read_csv(train_df_path)
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found in {dataframe_dir}")
    return train_df, val_df, test_df

def preprocess_metadata_csv(csv_path):
    """Load meta csv. Filter samples to be discarded."""
    meta_data = pd.read_csv(csv_path)
    meta_data = meta_data[meta_data.label_flair != 'discard']
    meta_data = meta_data[meta_data.label_t1 != 'discard']
    meta_data = meta_data[meta_data.label_t1ce != 'discard']
    meta_data = meta_data[meta_data.label_t2 != 'discard']
    meta_data_df  = meta_data.reset_index(drop =True)
    return meta_data_df

def get_data_from_csv(csv_path, split_exists, test_split, val_split, mod, output_dir, dataframe_dir):
    if split_exists:
        if  dataframe_dir is not None:
            train_df, val_df, test_df = get_splits(dataframe_dir) 
    else:
        meta_data_df = preprocess_metadata_csv(csv_path)
        # test_split = 0.4
        # val_split = 0.1
        train_df, val_df, test_df = split_dataset(meta_data_df, test_split, val_split, mod, output_dir)    
    return train_df, val_df, test_df

def get_CustomDataset_from_csv(df, mod, split):
    if split == 'train':
        data = CustomDataset_from_csv(df , mod,
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),    
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),        
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.0214, 0.0228, 0.0341], std=[0.2257, 0.2486, 0.2636]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # transforms.Normalize(mean=[0.0355, 0.0164, 0.0047], std=[0.2799, 0.1805, 0.1009]),(subset)            
            ])
            )
            #TODO: Calculate mean and stdD for .npy dataset to use for normalizing
    if split == 'test' or split == 'val':
        data = CustomDataset_from_csv(df , mod,
                transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    # transforms.Normalize(mean=[0.0214, 0.0228, 0.0341], std=[0.2257, 0.2486, 0.2636]),
            ])
            )
    return data

def get_dataloader(train_df, test_df, val_df, mod, batch_size):
    train_data = get_CustomDataset_from_csv(train_df, mod, 'train')
    test_data =get_CustomDataset_from_csv(test_df, mod, 'test')
    val_data = get_CustomDataset_from_csv(val_df, mod, 'val')
    print("Shape of trainig data :",train_data.dataframe.shape)
    print("Shape of validation data :",val_data.dataframe.shape)
    print("Shape of test data :",test_data.dataframe.shape)
    
    train_dataloader = DataLoader(train_data, batch_size = batch_size , shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = batch_size , shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size = batch_size , shuffle = False)
    return  train_dataloader, val_dataloader, test_dataloader

def get_multimodal_CustomDataset_from_csv(df, mod):
    data = CustomDataset_multimodal_from_csv(df , mod,
           transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=30),
            transforms.RandomRotation(degrees=270),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),    
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0214, 0.0228, 0.0341], std=[0.2257, 0.2486, 0.2636]),
            # transforms.Normalize(mean=[0.0355, 0.0164, 0.0047], std=[0.2799, 0.1805, 0.1009]),(subset)            
           ])
           )
           #TODO: Calculate mean and stdD for .npy dataset to use for normalizing
    return data

def get_multimodal_dataloader(train_df, test_df, val_df, batch_size): 
    train_data = get_multimodal_CustomDataset_from_csv(train_df, 'flair')
    test_data =get_multimodal_CustomDataset_from_csv(test_df, 'flair')
    val_data = get_multimodal_CustomDataset_from_csv(val_df, 'flair')
    print("Shape of trainig data :",train_data.dataframe.shape)
    print("Shape of validation data :",val_data.dataframe.shape)
    print("Shape of test data :",test_data.dataframe.shape)
    
    train_dataloader = DataLoader(train_data, batch_size = batch_size , shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = batch_size , shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = batch_size , shuffle = True)
    return  train_dataloader, val_dataloader, test_dataloader

